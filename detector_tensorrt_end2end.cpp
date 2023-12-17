#include "detector_tensorrt_end2end.h"
#include "fstream"
#include "NvInferPlugin.h"

#define CHECK(call)                                                         \
do {                                                                        \
        const cudaError_t error_code = call;                                \
        if (error_code != cudaSuccess) {                                    \
            printf("CUDA Error:\n");                                        \
            printf("    File:       %s\n", __FILE__);                       \
            printf("    Line:       %d\n", __LINE__);                       \
            printf("    Error code: %d\n", error_code);                     \
            printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
            exit(1);                                                        \
    }                                                                       \
} while (0)
inline int type_to_size(const nvinfer1::DataType& dataType)
{
    switch (dataType) {
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kINT8:
        return 1;
    case nvinfer1::DataType::kBOOL:
        return 1;
    default:
        return 4;
    }
}
inline int get_size_by_dims(const nvinfer1::Dims& dims)
{
    int size = 1;
    for (int i = 0; i < dims.nbDims; i++) {
        size *= dims.d[i];
    }
    return size;
}
inline static float clamp(float val, float min, float max)
{
    return val > min ? (val < max ? val : max) : min;
}

Detector_TensorRT_End2End::Detector_TensorRT_End2End(QObject *parent) : Detector{parent} {
    qDebug() << Q_FUNC_INFO;
}

Detector_TensorRT_End2End::~Detector_TensorRT_End2End()
{
    context->destroy();
    engine->destroy();
    runtime->destroy();
    cudaStreamDestroy(stream);
    for (auto& ptr : device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto& ptr : host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}

bool Detector_TensorRT_End2End::LoadModel(QString &modelPath)
{
    qDebug() << Q_FUNC_INFO;

    std::ifstream file(modelPath.toStdString(), std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    initLibNvInferPlugins(&gLogger, "");
    runtime = nvinfer1::createInferRuntime(gLogger);
    assert(runtime != nullptr);

    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    delete[] trtModelStream;
    context = engine->createExecutionContext();

    assert(context != nullptr);
    cudaStreamCreate(&stream);
    num_bindings = engine->getNbBindings();

    for (int i = 0; i < num_bindings; ++i) {
        Binding            binding;
        nvinfer1::Dims     dims;
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        std::string        name  = engine->getBindingName(i);
        binding.name             = name;
        binding.dsize            = type_to_size(dtype);

        bool IsInput = engine->bindingIsInput(i);
        if (IsInput) {
            num_inputs += 1;
            dims         = engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            input_bindings.push_back(binding);
            // set max opt shape
            context->setBindingDimensions(i, dims);
        }
        else {
            dims         = context->getBindingDimensions(i);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            output_bindings.push_back(binding);
            num_outputs += 1;
        }
    }
    // make_pipe
    for (auto& bindings : input_bindings) {
        void* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, stream));
        device_ptrs.push_back(d_ptr);
    }

    for (auto& bindings : output_bindings) {
        void * d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        device_ptrs.push_back(d_ptr);
        host_ptrs.push_back(h_ptr);
    }

    return true;
}

BatchDetectedObject Detector_TensorRT_End2End::Run(MatVector &srcImgList)
{
    qDebug() << Q_FUNC_INFO;

    // TODO: just work with bachNumber=1
    if(_batchSize > 1 || srcImgList.size() > 1) {
        qDebug() <<"This class just work with batchNumber=1";
        return {};
    }

    BatchDetectedObject batchOutput;
    ImagesDetectedObject imageOutput;

    auto srcImg = srcImgList[0];

    copy_from_Mat(srcImg, _inputSize);

    context->enqueueV2(device_ptrs.data(), stream, nullptr);
    for (int i = 0; i < num_outputs; i++) {
        size_t osize = output_bindings[i].size * output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(host_ptrs[i], device_ptrs[i + num_inputs], osize, cudaMemcpyDeviceToHost, stream));
    }
    cudaStreamSynchronize(stream);

    imageOutput = postprocess();

    batchOutput.push_back(imageOutput);
    return batchOutput;
}

void Detector_TensorRT_End2End::copy_from_Mat(const cv::Mat &image)
{
    cv::Mat  nchw;
    auto&    in_binding = input_bindings[0];
    auto     width      = in_binding.dims.d[3];
    auto     height     = in_binding.dims.d[2];
    cv::Size size{width, height};
    letterbox(image, nchw, size);

    context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, height, width}});

    CHECK(cudaMemcpyAsync(
        device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, stream));
}
void Detector_TensorRT_End2End::copy_from_Mat(const cv::Mat &image, cv::Size &size)
{
    cv::Mat nchw;
    letterbox(image, nchw, size);
    context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    CHECK(cudaMemcpyAsync(
        device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, stream));
}
void Detector_TensorRT_End2End::letterbox(const cv::Mat &image, cv::Mat &out, cv::Size &size)
{
    const float inp_h  = size.height;
    const float inp_w  = size.width;
    float       height = image.rows;
    float       width  = image.cols;

    float r    = std::min(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top    = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left   = int(std::round(dw - 0.1f));
    int right  = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
    pparam.ratio  = 1 / r;
    pparam.dw     = dw;
    pparam.dh     = dh;
    pparam.height = height;
    pparam.width  = width;
}
ImagesDetectedObject Detector_TensorRT_End2End::postprocess()
{
    ImagesDetectedObject imageOutput;

    int*  num_dets = static_cast<int*>(host_ptrs[0]);
    auto* boxes    = static_cast<float*>(host_ptrs[1]);
    auto* scores   = static_cast<float*>(host_ptrs[2]);
    int*  labels   = static_cast<int*>(host_ptrs[3]);
    auto& dw       = pparam.dw;
    auto& dh       = pparam.dh;
    auto& width    = pparam.width;
    auto& height   = pparam.height;
    auto& ratio    = pparam.ratio;
    for (int i = 0; i < num_dets[0]; i++) {
        float* ptr = boxes + i * 4;

        float x0 = *ptr++ - dw;
        float y0 = *ptr++ - dh;
        float x1 = *ptr++ - dw;
        float y1 = *ptr - dh;

        x0 = clamp(x0 * ratio, 0.f, width);
        y0 = clamp(y0 * ratio, 0.f, height);
        x1 = clamp(x1 * ratio, 0.f, width);
        y1 = clamp(y1 * ratio, 0.f, height);
        DetectedObject result;

        result.box.x = x0;
        result.box.y = y0;
        result.box.width = x1 - x0;
        result.box.height = y1 - y0;
        result.classID = *(labels + i);
        result.confidence = *(scores + i);

        imageOutput.push_back(result);
    }
    return imageOutput;
}
