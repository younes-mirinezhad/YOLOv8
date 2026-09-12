#include "enginebuilder_2.h"
#include <NvOnnxParser.h>
#include <fstream>

EngineBuilder_2::EngineBuilder_2() {}
EngineBuilder_2::~EngineBuilder_2()
{
    if (serializedModel) delete serializedModel;
    if (engine) delete engine;
    if (config) delete config;
    if (network) delete network;
    if (builder) delete builder;
}

bool EngineBuilder_2::buildEngine(const std::string &onnxFilePath, const std::string &engineFilePath)
{
    std::cout << "----- Start initializings ..." << std::endl;

    std::cout << "----- Create logger ..." << std::endl;
    Logger_2 gLogger;

    std::cout << "----- Create builder ..." << std::endl;
    builder = nvinfer1::createInferBuilder(gLogger);

    std::cout << "----- Create network ..." << std::endl;
    auto flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    network = builder->createNetworkV2(flag);

    std::cout << "----- Create ONNX parser ..." << std::endl;
    auto parser = nvonnxparser::createParser(*network, gLogger);
    if (!parser->parseFromFile(onnxFilePath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "***** Failed to parse ONNX file." << std::endl;
        return false;
    }

    addPlugin(network);

    std::cout << "----- Create builder config ..." << std::endl;
    config = builder->createBuilderConfig();

    std::cout << "----- Start building engine file ..." << std::endl;
    engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine) {
        std::cerr << "***** Failed to build engine." << std::endl;
        return false;
    }

    std::cout << "----- Start writing engine file ..." << std::endl;
    serializedModel = engine->serialize();
    std::ofstream engineFile(engineFilePath, std::ios::binary);
    engineFile.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
    engineFile.close();
    std::cout << "---------- Writing engine finished." << std::endl;

    return true;
}

void EngineBuilder_2::addPlugin(nvinfer1::INetworkDefinition *network)
{
    auto previousOutput = network->getOutput(0);
    network->unmarkOutput(*previousOutput);

    // Define the dimensions for slicing
    auto strides = nvinfer1::Dims3(1, 1, 1);
    auto starts = nvinfer1::Dims3(0, 0, 0);

    // Slice the tensor to separate boxes and scores
    auto shape = previousOutput->getDimensions();

    shape.d[1] = 4; // First 4 values are bounding box coordinates
    auto boxes = network->addSlice(*previousOutput, starts, shape, strides);

    shape.d[1] = 80; // Next 80 values are class confidences
    starts.d[1] = 4; // Start slicing from the 5th element
    auto scores = network->addSlice(*previousOutput, starts, shape, strides);

    // Define plugin fields
    int pluginVariable[] = { _backGroundClass };
    auto bgPf = nvinfer1::PluginField("background_class", &pluginVariable[0], nvinfer1::PluginFieldType::kINT32);

    int pluginVariable1[] = { _maxNumOfDetection };
    auto maxOutputBoxesPf = nvinfer1::PluginField("max_output_boxes", &pluginVariable1[0], nvinfer1::PluginFieldType::kINT32);

    int pluginVariable2[] = { _scoreActivation };
    auto scoreActivationPf = nvinfer1::PluginField("score_activation", &pluginVariable2[0], nvinfer1::PluginFieldType::kINT32);

    int pluginVariable3[] = { _boxCoding };
    auto boxEncodingPf = nvinfer1::PluginField("box_coding", &pluginVariable3[0], nvinfer1::PluginFieldType::kINT32);

    float pluginFVariable1[] = { _confidenceThreshold };
    auto scoreThPf = nvinfer1::PluginField("score_threshold", &pluginFVariable1[0], nvinfer1::PluginFieldType::kFLOAT32);

    float pluginFVariable12[] = { _iouThreshold };
    auto iouThPf = nvinfer1::PluginField("iou_threshold", &pluginFVariable12[0], nvinfer1::PluginFieldType::kFLOAT32);

    std::vector<nvinfer1::PluginField> fc = { bgPf, maxOutputBoxesPf, scoreThPf, iouThPf, boxEncodingPf, scoreActivationPf };
    const nvinfer1::PluginFieldCollection pluginData = { 6, fc.data() };

    nvinfer1::EngineCapability capability;
    auto registry = nvinfer1::getBuilderPluginRegistry(capability);
    assert(registry != nullptr && "registry is null pointer");

    auto creator = registry->getPluginCreator("EfficientNMS_TRT", "1");
    auto nmsLayer = creator->createPlugin("nms_layer", &pluginData);

    std::vector<nvinfer1::ITensor *> outs;
    outs.push_back(boxes->getOutput(0));
    outs.push_back(scores->getOutput(0));

    auto layer = network->addPluginV2(outs.data(), outs.size(), *nmsLayer);
    layer->getOutput(0)->setName("num");
    layer->getOutput(1)->setName("boxes");
    layer->getOutput(2)->setName("scores");
    layer->getOutput(3)->setName("classes");

    for (int i = 0; i < 4; i++) {
        network->markOutput(*layer->getOutput(i));
        std::cout << "layer name: " << layer->getOutput(i)->getName() << "shape " << getTensorShape(layer->getOutput(i)) << std::endl;
    }
}

std::string EngineBuilder_2::getTensorShape(nvinfer1::ITensor *tensor)
{
    std::string shape{};
    for (int j = 0; j < tensor->getDimensions().nbDims; j++) {

        shape += std::to_string(tensor->getDimensions().d[j]) + (j < tensor->getDimensions().nbDims - 1 ? " x " : "");
    }
    return shape;
}
