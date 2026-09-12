#include "detector_tensorrt.h"
#include <iostream>
#include <fstream>
#include "NvInferPlugin.h"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/imgcodecs.hpp>

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

void Logger::setLogSeverity(Severity severity)
{
    m_severity = severity;
}
void Logger::log(Severity severity, const char *msg) noexcept
{
    if (severity <= m_severity) {
        switch (severity) {
        case Severity::kINTERNAL_ERROR:
            std::cout << "[F] [TRT] -----> " << msg << std::endl;
            break;
        case Severity::kERROR:
            std::cout << "[E] [TRT] -----> " << msg << std::endl;
            break;
        case Severity::kWARNING:
            std::cout << "[W] [TRT] -----> " << msg << std::endl;
            break;
        case Severity::kINFO:
            std::cout << "[I] [TRT] -----> " << msg << std::endl;
            break;
        case Severity::kVERBOSE:
            std::cout << "[V] [TRT] -----> " << msg << std::endl;
            break;
        default:
            std::cerr << "Invalid log level -----> " << msg << std::endl;
            assert(false && "Invalid log level");
            break;
        }
    }
}

Detector_TensorRT::Detector_TensorRT() {}
Detector_TensorRT::~Detector_TensorRT()
{
    delete context;
    // Synchronize and destroy the cuda stream
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaStreamDestroy(stream));
    for (auto& ptr : device_ptrs) {
        CHECK(cudaFree(ptr));
    }
    for (auto& ptr : host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}

void Detector_TensorRT::setInputSize(cv::Size newInputSize)
{
    _inputSize = newInputSize;
}

void Detector_TensorRT::setClassNames(QStringList newClassNames)
{
    _classNames.clear();
    _classNames = newClassNames;
}

bool Detector_TensorRT::loadModel(std::string &modelPath)
{
    std::ifstream file(modelPath, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    Logger gLogger;
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
        const char *name = engine->getIOTensorName(i);
        auto dtype = engine->getTensorDataType(name);
        binding.name = name;
        binding.dsize = type_to_size(dtype);

        bool IsInput = engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        // if(input_bindings.size() > 0) std::cout << input_bindings[0].name << std::endl;

        if (IsInput) {
            num_inputs += 1;
            dims = engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            // std::cout << binding.name << std::endl;

            input_bindings.push_back(binding);
            context->setInputShape(name, dims);
        } else {
            num_outputs += 1;
            dims = context->getTensorShape(name);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            output_bindings.push_back(binding);
            // std::cout << binding.name << std::endl;
        }
    }
    // make_pipe
    for (auto& bindings : input_bindings) {
        // std::cout << bindings.name << std::endl;

        void* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, stream));
        context->setTensorAddress(bindings.name, d_ptr);
        device_ptrs.push_back(d_ptr);
    }

    for (auto& bindings : output_bindings) {
        void * d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        context->setTensorAddress(bindings.name, d_ptr);
        device_ptrs.push_back(d_ptr);
        host_ptrs.push_back(h_ptr);
    }
    CHECK(cudaStreamSynchronize(stream));

    _modelIsLoaded = true;
    return _modelIsLoaded;
}

QList<QList<DetectedObject>> Detector_TensorRT::detect(std::vector<cv::Mat> &src_imgs)
{
    _batch = src_imgs.size();
    add_pad = false;
    cv::cuda::GpuMat model_input;
    if(use_gpumat){
        std::vector<cv::cuda::GpuMat> g_images;
        for (auto &img : src_imgs) {
            cv::cuda::GpuMat gImg;
            gImg.upload(img);
            g_images.push_back(gImg);
        }

        model_input = preprocess(g_images);
    }
    else{
        return{};
    }

    inference(model_input);

    auto dets = postprocess();

    return dets;
}

inline static float clamp(float val, float min, float max) { return val > min ? (val < max ? val : max) : min; }

cv::cuda::GpuMat Detector_TensorRT::preprocess(std::vector<cv::cuda::GpuMat> &src_imgs)
{
    for (cv::cuda::GpuMat &image : src_imgs) {
        cv::cuda::cvtColor(image, image, cv::COLOR_BGR2RGB);

        float height = image.rows;
        float width  = image.cols;

        float r      = std::min(_inputSize.height / height, _inputSize.width / width);
        int   padw   = std::round(width * r);
        int   padh   = std::round(height * r);

        if ((int)width != padw || (int)height != padh) {
            cv::cuda::resize(image, image, cv::Size(padw, padh));
        }

        float dw    = _inputSize.width - padw;
        float dh    = _inputSize.height - padh;

        dw         /= 2.0f;
        dh         /= 2.0f;
        int top     = int(std::round(dh - 0.1f));
        int bottom  = int(std::round(dh + 0.1f));
        int left    = int(std::round(dw - 0.1f));
        int right   = int(std::round(dw + 0.1f));

        cv::cuda::copyMakeBorder(image, image, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});
        // image.convertTo(image, CV_32FC3, 1.0f / 255.0f);

        top_pad       = top;
        left_pad      = left;
        add_pad       = true;

        PreParam pparam;
        pparam.ratio  = 1 / r;
        pparam.dw     = dw;
        pparam.dh     = dh;
        pparam.height = height;
        pparam.width  = width;

        pparams.append(pparam);
    }

    auto gblob = blob_from_GpuMats(src_imgs);

    return gblob;
}

cv::cuda::GpuMat Detector_TensorRT::blob_from_GpuMats(std::vector<cv::cuda::GpuMat> &batchInput)
{
    cv::cuda::GpuMat gpu_dst(1, batchInput[0].rows * batchInput[0].cols * batchInput.size(), CV_8UC3);

    size_t width = batchInput[0].cols * batchInput[0].rows;
    for (size_t img = 0; img < batchInput.size(); img++) {
        std::vector<cv::cuda::GpuMat> input_channels{
            cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[0 + width * 3 * img])),
            cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width + width * 3 * img])),
            cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width * 2 + width * 3 * img]))
        };
        cv::cuda::split(batchInput[img], input_channels);  // HWC -> CHW
    }
    cv::cuda::GpuMat mfloat;

    gpu_dst.convertTo(mfloat, CV_32FC3, 1.f / 255.f);

    return mfloat;
}

void Detector_TensorRT::inference(cv::cuda::GpuMat &model_input)
{
    int total = model_input.rows * model_input.cols;

    CHECK(cudaMemcpyAsync(device_ptrs[0], model_input.ptr<float>(), total * model_input.elemSize(), cudaMemcpyHostToDevice, stream));

    // Ensure all dynamic bindings have been defined.
    if (!context->allInputDimensionsSpecified()) {
        throw std::runtime_error("Error, not all required dimensions specified.");
    }
    context->enqueueV3(stream);
    for (int i = 0; i < num_outputs; i++) {
        size_t osize = output_bindings[i].size * output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(host_ptrs[i], device_ptrs[i + num_inputs], osize, cudaMemcpyDeviceToHost, stream));
    }
    CHECK(cudaStreamSynchronize(stream));
}

QList<QList<DetectedObject>> Detector_TensorRT::postprocess()
{
    int* num_dets = static_cast<int*>(host_ptrs[0]);
    auto* boxes = static_cast<float*>(host_ptrs[1]);
    auto* scores = static_cast<float*>(host_ptrs[2]);
    int* labels = static_cast<int*>(host_ptrs[3]);

    QList<QList<DetectedObject>> detections;

    for (int i = 0; i < _batch; i++) {
        QList<DetectedObject> dets;
        auto pparam = pparams[i];
        auto& dw = pparam.dw;
        auto& dh = pparam.dh;
        auto& width = pparam.width;
        auto& height = pparam.height;
        auto& ratio = pparam.ratio;
        for (int j = 0; j < num_dets[i]; j++) {
            float conf = *(scores + j);
            if(conf < 0.1) continue;
            float* ptr = boxes + j * 4;
            float x0 = *ptr++;
            float y0 = *ptr++;
            float x1 = *ptr++;
            float y1 = *ptr;

            x0 = x0 - dw;
            y0 = y0 - dh;
            x1 = x1 - dw;
            y1 = y1 - dh;

            x0 = clamp(x0 * ratio, 0.f, width);
            y0 = clamp(y0 * ratio, 0.f, height);
            x1 = clamp(x1 * ratio, 0.f, width);
            y1 = clamp(y1 * ratio, 0.f, height);

            DetectedObject det;

            det.box.x = x0;
            det.box.y = y0;
            det.box.width = x1 - x0;
            det.box.height = y1 - y0;
            det.classID = *(labels + j);
            det.className = _classNames[det.classID];
            det.confidence = conf;

            dets.push_back(det);
        }

        detections.push_back(dets);
        boxes += 100*4;
        scores += 100;
        labels += 100;
    }

    return detections;
}
