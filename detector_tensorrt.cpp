#include "detector_tensorrt.h"
#include "qdebug.h"
#include "fstream"
#include "NvInferPlugin.h"
#include <opencv2/cudawarping.hpp>
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"

#define CHECK(call)\
do {\
        const cudaError_t error_code = call;\
        if (error_code != cudaSuccess) {\
            printf("CUDA Error:\n");\
            printf("    File:       %s\n", __FILE__);\
            printf("    Line:       %d\n", __LINE__);\
            printf("    Error code: %d\n", error_code);\
            printf("    Error text: %s\n", cudaGetErrorString(error_code));\
            exit(1);\
    }\
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
    for (int i = 0; i < dims.nbDims; i++)
        size *= dims.d[i];
    return size;
}

Detector_TensorRT::Detector_TensorRT(QObject *parent) : Detector{parent} {}
Detector_TensorRT::~Detector_TensorRT()
{
    context->destroy();
    CHECK(cudaStreamDestroy(stream));
    for (auto& ptr : device_ptrs) {
        CHECK(cudaFree(ptr));
    }
    for (auto& ptr : host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}

bool Detector_TensorRT::LoadModel(QString &modelPath)
{
    qDebug() << Q_FUNC_INFO << modelPath;

    std::ifstream file(modelPath.toStdString(), std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    Logger gLogger{nvinfer1::ILogger::Severity::kERROR};
    initLibNvInferPlugins(&gLogger, "");
    nvinfer1::IRuntime* runtime = nullptr;
    runtime = nvinfer1::createInferRuntime(gLogger);
    assert(runtime != nullptr);

    nvinfer1::ICudaEngine* engine = nullptr;
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    delete[] trtModelStream;
    context = engine->createExecutionContext();

    assert(context != nullptr);
    CHECK(cudaStreamCreate(&stream));
    int num_bindings = engine->getNbIOTensors();

    for (int i = 0; i < num_bindings; ++i) {
        Binding binding;
        nvinfer1::Dims dims;
        std::string name = engine->getIOTensorName(i);
        auto dtype = engine->getTensorDataType(name.data());
        binding.name = name.data();
        binding.dsize = type_to_size(dtype);

        bool IsInput = engine->bindingIsInput(i);
        if (IsInput) {
            num_inputs += 1;
            dims = engine->getProfileShape(name.data(), 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            input_bindings.push_back(binding);
            // set max opt shape
            context->setInputShape(name.data(), dims);
        } else {
            num_outputs += 1;
            dims = context->getTensorShape(name.data());
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            output_bindings.push_back(binding);
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
    CHECK(cudaStreamSynchronize(stream));

    qDebug() << "----- Model loaded:" << modelPath;

    _gBlob = cv::cuda::GpuMat(1, _inputSize.width * _inputSize.height, CV_32FC3);
    _blob = cv::Mat(1, _inputSize.width * _inputSize.height, CV_32FC3);

    return true;
}

ImagesDetectedObject Detector_TensorRT::detect(cv::Mat &srcImg)
{
    copy_from_Mat(srcImg, _inputSize);
    context->enqueueV2(device_ptrs.data(), stream, nullptr);
    for (int i = 0; i < num_outputs; i++) {
        size_t osize = output_bindings[i].size * output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(host_ptrs[i], device_ptrs[i + num_inputs], osize, cudaMemcpyDeviceToHost, stream));
    }
    CHECK(cudaStreamSynchronize(stream));
    auto resOneShot = postprocess();

    return resOneShot;
}
void Detector_TensorRT::copy_from_Mat(cv::Mat &img, cv::Size &size)
{
    letterbox(img, size);
    context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});

    CHECK(cudaMemcpyAsync(device_ptrs[0], _blob.ptr<float>(), _blob.total() * _blob.elemSize(), cudaMemcpyHostToDevice, stream));
}
void Detector_TensorRT::letterbox(cv::Mat &img_input, cv::Size &size)
{
    const float inp_h = size.height;
    const float inp_w = size.width;
    float height = img_input.rows;
    float width = img_input.cols;

    float r = std::min(inp_h / height, inp_w / width);
    int padw = std::round(width * r);
    int padh = std::round(height * r);

    if ((int)width != padw || (int)height != padh) {
        cv::resize(img_input, img_input, cv::Size(padw, padh));
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(img_input, img_input, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    cv::dnn::blobFromImage(img_input, _blob, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
    pparam.ratio = 1 / r;
    pparam.dw = dw;
    pparam.dh = dh;
    pparam.height = height;
    pparam.width = width;
}

ImagesDetectedObject Detector_TensorRT::detect(cv::cuda::GpuMat &srcImg)
{
    copy_from_Mat(srcImg, _inputSize);
    context->enqueueV2(device_ptrs.data(), stream, nullptr);
    for (int i = 0; i < num_outputs; i++) {
        size_t osize = output_bindings[i].size * output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(host_ptrs[i], device_ptrs[i + num_inputs], osize, cudaMemcpyDeviceToHost, stream));
    }
    CHECK(cudaStreamSynchronize(stream));
    auto resOneShot = postprocess();

    return resOneShot;
}
void Detector_TensorRT::copy_from_Mat(cv::cuda::GpuMat &gImg, cv::Size &size)
{
    letterbox(gImg, size);

    context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    int total = _gBlob.rows * _gBlob.cols;
    CHECK(cudaMemcpyAsync(device_ptrs[0], _gBlob.ptr<float>(), total * _gBlob.elemSize(), cudaMemcpyHostToDevice, stream));
}
void Detector_TensorRT::letterbox(cv::cuda::GpuMat &gImg_input, cv::Size &size)
{
    const float inp_h = size.height;
    const float inp_w = size.width;
    float height = gImg_input.rows;
    float width = gImg_input.cols;

    float r = std::min(inp_h / height, inp_w / width);
    int padw = std::round(width * r);
    int padh = std::round(height * r);

    if ((int)width != padw || (int)height != padh) {
        cv::cuda::resize(gImg_input, gImg_input, cv::Size(padw, padh));
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));

    cv::cuda::copyMakeBorder(gImg_input, gImg_input, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});
    blobFromGpuMat(gImg_input, std::array<float, 3>{1, 1, 1}, std::array<float, 3>{0, 0, 0}, false, true);

    pparam.ratio = 1 / r;
    pparam.dw = dw;
    pparam.dh = dh;
    pparam.height = height;
    pparam.width = width;
}
void Detector_TensorRT::blobFromGpuMat(const cv::cuda::GpuMat &gImg_input, const std::array<float, 3> &std, const std::array<float, 3> &mean, bool swapBR, bool normalize)
{
    if (swapBR)
        cv::cuda::cvtColor(gImg_input, gImg_input, cv::COLOR_BGR2RGB);

    // if using global blob, it needed to use convertTo(CV_8UC3)
    cv::cuda::GpuMat blob(1, gImg_input.rows * gImg_input.cols, CV_8UC3);

    size_t continuous_length = gImg_input.rows * gImg_input.cols;
    std::vector<cv::cuda::GpuMat> rgb {
        cv::cuda::GpuMat(gImg_input.rows, gImg_input.cols, CV_8U, &(blob.ptr()[0])),
        cv::cuda::GpuMat(gImg_input.rows, gImg_input.cols, CV_8U, &(blob.ptr()[continuous_length])),
        cv::cuda::GpuMat(gImg_input.rows, gImg_input.cols, CV_8U, &(blob.ptr()[continuous_length * 2])),
    };
    cv::cuda::split(gImg_input, rgb);
    if (normalize) {
        blob.convertTo(blob, CV_32FC3, 1.f / 255.f);
    } else {
        blob.convertTo(blob, CV_32FC3);
    }
    cv::cuda::subtract(blob, cv::Scalar(mean[0], mean[1], mean[2]), blob, cv::noArray(), -1);
    cv::cuda::divide(blob, cv::Scalar(std[0], std[1], std[2]), blob, 1, -1);

    _gBlob = blob;
}

inline static float clamp(float val, float min, float max)
{
    return val > min ? (val < max ? val : max) : min;
}
ImagesDetectedObject Detector_TensorRT::postprocess()
{
    ImagesDetectedObject res_OneShot;

    int* num_dets = static_cast<int*>(host_ptrs[0]);
    auto* boxes = static_cast<float*>(host_ptrs[1]);
    auto* scores = static_cast<float*>(host_ptrs[2]);
    int* labels = static_cast<int*>(host_ptrs[3]);
    auto& dw = pparam.dw;
    auto& dh = pparam.dh;
    auto& width = pparam.width;
    auto& height = pparam.height;
    auto& ratio = pparam.ratio;
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
        result.className = _classNamesList[result.classID];
        result.confidence = *(scores + i);

        res_OneShot.push_back(result);
    }

    return res_OneShot;
}
