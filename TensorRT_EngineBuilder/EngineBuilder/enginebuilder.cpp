#include "enginebuilder.h"
#include "spdlog/spdlog.h"
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
            spdlog::critical("[F] [TRT] {} -----> {}", Q_FUNC_INFO, msg);
            break;
        case Severity::kERROR:
            spdlog::error("[E] [TRT] {} -----> {}", Q_FUNC_INFO, msg);
            break;
        case Severity::kWARNING:
            spdlog::warn("[W] [TRT] {} -----> {}", Q_FUNC_INFO, msg);
            break;
        case Severity::kINFO:
            spdlog::info("[I] [TRT] {} -----> {}", Q_FUNC_INFO, msg);
            break;
        case Severity::kVERBOSE:
            spdlog::info("[V] [TRT] {} -----> {}", Q_FUNC_INFO, msg);
            break;
        default:
            assert(false && "{} -----> invalid log level");
            break;
        }
    }
}

EngineBuilder::EngineBuilder(const std::string &onnxPath, const std::string &enginePath, QObject *parent)
{
    spdlog::info("{} -----> Start initializings ...", Q_FUNC_INFO);

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

    spdlog::info("{} -----> End initializings.", Q_FUNC_INFO);
}
EngineBuilder::~EngineBuilder() { }

void EngineBuilder::buildEngine()
{
    spdlog::info("{} -----> Start building engine from {} ...", Q_FUNC_INFO, _onnxPath);

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

    spdlog::info("{} ----------> Network description:", Q_FUNC_INFO);

    for (auto &tensor : inputs) {
        auto shape = getTensorShape(tensor);
        spdlog::info("{} ---------------> Input name : {}, shape: {}", Q_FUNC_INFO, tensor->getName(), shape);
    }

    for (auto &tensor : outputs) {
        auto shape = getTensorShape(tensor);
        spdlog::info("{} ---------------> Output name : {}, shape: {}", Q_FUNC_INFO, tensor->getName(), shape);
    }

    writeEngine();

    spdlog::info("{} -----> End building.", Q_FUNC_INFO);
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
    spdlog::info("{} -----> Start writing engine ...", Q_FUNC_INFO);

    if (!_builder->platformHasFastFp16()) {
        spdlog::error("{} ********** FP16 is not supported.", Q_FUNC_INFO);
    }
    _config->setFlag(nvinfer1::BuilderFlag::kFP16);
    auto f = _builder->buildSerializedNetwork(*_network, *_config);

    std::ofstream file;
    file.open(_enginePath, std::ios::binary | std::ios::out);
    if (!file.is_open()) {
        spdlog::error("{} ********** create engine file failed.", Q_FUNC_INFO);
        return;
    }
    file.write((const char *)f->data(), f->size());
    file.close();

    spdlog::info("{} ----------> Engine saved to {}", Q_FUNC_INFO, _enginePath);
    spdlog::info("{} -----> End writing engine.", Q_FUNC_INFO);
}
