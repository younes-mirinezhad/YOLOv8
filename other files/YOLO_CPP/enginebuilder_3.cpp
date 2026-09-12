#include "enginebuilder_3.h"
#include <NvOnnxParser.h>
#include <fstream>

EngineBuilder_3::EngineBuilder_3() {}

inline void checkCudaErrorCode(cudaError_t code) {
    if (code != cudaSuccess) {
        std::string errMsg = "CUDA operation failed with code: " + std::to_string(code)
                             + " (" + cudaGetErrorName(code) + "), with message: " + cudaGetErrorString(code);
        std::cerr << errMsg << std::endl;
        throw std::runtime_error(errMsg);
    }
}
bool EngineBuilder_3::build(std::string onnxModelPath, std::string enginePath)
{
    Logger_3 m_logger;
    int optBatchSize = 1;
    int maxBatchSize = 1;
    auto precision = "FP16";
    std::string calibrationDataDirectoryPath = "";

    // Create our engine builder.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) {
        return false;
    }

    // Define an explicit batch size and then create the network (implicit batch
    // size is deprecated). More info here:
    // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
    auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        return false;
    }

    // Create a parser for reading the onnx file.
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if (!parser) {
        return false;
    }

    // We are going to first read the onnx file into memory, then pass that buffer
    // to the parser. Had our onnx model file been encrypted, this approach would
    // allow us to first decrypt the buffer.
    std::ifstream file(onnxModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        auto msg = "Error, unable to read engine file";
        std::cerr << msg << std::endl;
        throw std::runtime_error(msg);
    }

    // Parse the buffer we read into memory.
    auto parsed = parser->parse(buffer.data(), buffer.size());
    if (!parsed) {
        return false;
    }

    // Ensure that all the inputs have the same batch size
    const auto numInputs = network->getNbInputs();
    if (numInputs < 1) {
        auto msg = "Error, model needs at least 1 input!";
        std::cerr << msg << std::endl;
        throw std::runtime_error(msg);
    }
    const auto input0Batch = network->getInput(0)->getDimensions().d[0];
    for (int32_t i = 1; i < numInputs; ++i) {
        if (network->getInput(i)->getDimensions().d[0] != input0Batch) {
            auto msg = "Error, the model has multiple inputs, each with differing batch sizes!";
            std::cerr << msg << std::endl;
            throw std::runtime_error(msg);
        }
    }

    // Check to see if the model supports dynamic batch size or not
    bool doesSupportDynamicBatch = false;
    if (input0Batch == -1) {
        doesSupportDynamicBatch = true;
        std::cout << "Model supports dynamic batch size" << std::endl;
    } else {
        std::cout << "Model only supports fixed batch size of " << input0Batch << std::endl;
        // If the model supports a fixed batch size, ensure that the maxBatchSize
        // and optBatchSize were set correctly.
        if (optBatchSize != input0Batch || maxBatchSize != input0Batch) {
            auto msg = "Error, model only supports a fixed batch size of " + std::to_string(input0Batch) +
                       ". Must set Options.optBatchSize and Options.maxBatchSize to 1";
            std::cerr << msg << std::endl;
            throw std::runtime_error(msg);
        }
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    // Register a single optimization profile
    nvinfer1::IOptimizationProfile *optProfile = builder->createOptimizationProfile();
    for (int32_t i = 0; i < numInputs; ++i) {
        // Must specify dimensions for all the inputs the model expects.
        const auto input = network->getInput(i);
        const auto inputName = input->getName();
        const auto inputDims = input->getDimensions();
        int32_t inputC = inputDims.d[1];
        int32_t inputH = inputDims.d[2];
        int32_t inputW = inputDims.d[3];

        // Specify the optimization profile`
        if (doesSupportDynamicBatch) {
            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, inputC, inputH, inputW));
        } else {
            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(optBatchSize, inputC, inputH, inputW));
        }
        optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(optBatchSize, inputC, inputH, inputW));
        optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(maxBatchSize, inputC, inputH, inputW));
    }
    config->addOptimizationProfile(optProfile);

    // Set the precision level
    if (precision == "FP16") {
        // Ensure the GPU supports FP16 inference
        if (!builder->platformHasFastFp16()) {
            auto msg = "Error: GPU does not support FP16 precision";
            std::cerr << msg << std::endl;
            throw std::runtime_error(msg);
        }
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // CUDA stream used for profiling by the builder.
    cudaStream_t profileStream;
    checkCudaErrorCode(cudaStreamCreate(&profileStream));
    config->setProfileStream(profileStream);

    // Build the engine
    // If this call fails, it is suggested to increase the logger verbosity to
    // kVERBOSE and try rebuilding the engine. Doing so will provide you with more
    // information on why exactly it is failing.
    std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }

    addPlugin();

    // Write the engine to disk
    std::ofstream outfile(enginePath, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char *>(plan->data()), plan->size());
    std::cerr << "Success, saved engine to " << enginePath << std::endl;

    checkCudaErrorCode(cudaStreamDestroy(profileStream));
    return true;
}

void EngineBuilder_3::addPlugin()
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

std::string EngineBuilder_3::getTensorShape(nvinfer1::ITensor *tensor)
{
    std::string shape{};
    for (int j = 0; j < tensor->getDimensions().nbDims; j++) {

        shape += std::to_string(tensor->getDimensions().d[j]) + (j < tensor->getDimensions().nbDims - 1 ? " x " : "");
    }
    return shape;
}
