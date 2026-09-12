#include "enginebuilder.h"
#include <iostream>
#include <qdebug.h>
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
#include <fstream>

void TrtLogger::setLogSeverity(Severity severity)
{
    m_severity = severity;
}
void TrtLogger::log(Severity severity, const char *msg) noexcept
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

EngineBuilder::EngineBuilder(const std::string &onnxPath, const std::string &enginePath, QObject *parent)
{
    qDebug() << Q_FUNC_INFO << " -----> Start initializings ...";

    _onnxPath = onnxPath;
    _enginePath = enginePath;

    _logger.reset(new TrtLogger());
    assert(_logger != nullptr && "create trt builder failed");

    initLibNvInferPlugins(_logger.get(), "");
    _builder.reset(nvinfer1::createInferBuilder(*_logger));

    assert(_builder != nullptr && "create trt builder failed");
    _config.reset(_builder->createBuilderConfig());
    assert(_config != nullptr && "create trt builder config failed");
#if !(NV_TENSORRT_MAJOR >= 8 && NV_TENSORRT_MINOR >= 4)
    mConfig->setMaxWorkspaceSize(1 << 30); // 1GB
#endif
    // m_config->setMaxWorkspaceSize(1 << 30);
    _profile = _builder->createOptimizationProfile();
    assert(_profile != nullptr && "create trt builder optimazation profile failed");

    qDebug() << Q_FUNC_INFO << " -----> End initializings.";
}
EngineBuilder::~EngineBuilder() {}

void EngineBuilder::buildEngine()
{
    qDebug() << Q_FUNC_INFO << " -----> Start building engine from: " << _onnxPath;

    TrtUniquePtr<nvinfer1::IRuntime> runtime{ nvinfer1::createInferRuntime(*_logger) };
    assert(runtime != nullptr && "create trt runtime failed");
    auto flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    _network.reset(_builder->createNetworkV2(flag));
    assert(_network != nullptr && "create trt network failed");

    TrtUniquePtr<nvonnxparser::IParser> parser{ nvonnxparser::createParser(*_network, *_logger) };
    assert(parser != nullptr && "create trt onnx parser failed");
    bool parse_success = parser->parseFromFile(_onnxPath.c_str(), static_cast<int>(Severity::kWARNING));

    assert(parse_success && "parse onnx file failed");

    // get the input tensor
    std::vector<nvinfer1::ITensor *> inputs{};
    for (int i{ 0 }; i < _network->getNbInputs(); i++) {
        auto tensor = _network->getInput(i);
        inputs.push_back(tensor);
    }

    // get the out tensor
    std::vector<nvinfer1::ITensor *> outputs{};
    for (int i{ 0 }; i < _network->getNbOutputs(); i++) {
        auto tensor = _network->getOutput(i);
        outputs.push_back(tensor);
    }

    qDebug() << Q_FUNC_INFO << " ----------> Network description:";

    for (auto &tensor : inputs) {
        auto shape = getTensorShape(tensor);
        qDebug() << Q_FUNC_INFO << " ---------------> Input name: " << tensor->getName() << " , shape: " << shape;
    }

    for (auto &tensor : outputs) {
        auto shape = getTensorShape(tensor);
        qDebug() << Q_FUNC_INFO << " ---------------> Output name: " << tensor->getName() << " , shape: " << shape;
    }

    writeEngine();

    qDebug() << Q_FUNC_INFO << " -----> End building.";
}

std::string EngineBuilder::getTensorShape(nvinfer1::ITensor *tensor)
{
    std::string shape{};
    for (int j = 0; j < tensor->getDimensions().nbDims; j++) {
        shape += std::to_string(tensor->getDimensions().d[j]) + (j < tensor->getDimensions().nbDims - 1 ? " x " : "");
    }
    return shape;
}

void EngineBuilder::writeEngine()
{
    qDebug() << Q_FUNC_INFO << " -----> Start writing engine ...";

    if (!_builder->platformHasFastFp16()) {
        qDebug() << Q_FUNC_INFO << " ********** FP16 is not supported.";
    }
    _config->setFlag(nvinfer1::BuilderFlag::kFP16);
    auto f = _builder->buildSerializedNetwork(*_network, *_config);

    std::ofstream file;
    file.open(_enginePath, std::ios::binary | std::ios::out);
    if (!file.is_open()) {
        qDebug() << Q_FUNC_INFO << " ********** create engine file failed.";
        return;
    }
    file.write((const char *)f->data(), f->size());
    file.close();

    qDebug() << Q_FUNC_INFO << " ----------> Engine saved to: " << _enginePath;
    qDebug() << Q_FUNC_INFO << " -----> End writing engine.";
}
