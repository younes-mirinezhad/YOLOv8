#ifndef ENGINEBUILDER_H
#define ENGINEBUILDER_H

#include <QObject>
#include <NvInfer.h>

template <typename T> struct TrtDestroyer { void operator()(T *t) { delete t; } };
template <typename T> using TrtUniquePtr = std::unique_ptr<T, TrtDestroyer<T>>;
using Severity = nvinfer1::ILogger::Severity;

class TrtLogger : public nvinfer1::ILogger {
public:
    void setLogSeverity(Severity severity);
private:
    void log(Severity severity, const char *msg) noexcept override;
    Severity m_severity = Severity::kINFO;
};

class EngineBuilder
{
public:
    EngineBuilder(const std::string &onnxPath, const std::string &enginePath, QObject *parent = nullptr);
    ~EngineBuilder();

    void buildEngine();

private:
    std::string _onnxPath;
    std::string _enginePath;
    std::unique_ptr<TrtLogger> _logger{ nullptr };
    TrtUniquePtr<nvinfer1::IBuilder> _builder{ nullptr };
    TrtUniquePtr<nvinfer1::IBuilderConfig> _config{ nullptr };
    nvinfer1::IOptimizationProfile *_profile = nullptr;
    TrtUniquePtr<nvinfer1::INetworkDefinition> _network{ nullptr };
    std::string getTensorShape(nvinfer1::ITensor *tensor);
    void writeEngine();
};

#endif // ENGINEBUILDER_H
