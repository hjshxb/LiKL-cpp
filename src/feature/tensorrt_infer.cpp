#include "likl/feature/tensorrt_infer.h"
#include "likl/third_party/tensorrtbuffer/logger.h"


#ifdef WITH_TENSORRT

using namespace tensorrt_common;
using namespace tensorrt_log;
using namespace tensorrt_buffer;

namespace likl {


bool TensorRTInference::Build() {
    if (LoadEngine()) {
        return true;
    }
    auto builder = TensorRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder) {
        return false;
    }
    const auto explicit_batch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TensorRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
    if (!network) {
        return false;
    }
    auto config = TensorRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }
    auto parser = TensorRTUniquePtr<nvonnxparser::IParser>(
            nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser) {
        return false;
    }
    
    
    auto constructed = ConstructNetwork(builder, network, config, parser);
    if (!constructed) {
        return false;
    }

    nvinfer1::Dims dim = network->getInput(0)->getDimensions();
    if (dim.d[0] == -1) {
        // -1 means it is a dynamic model 
        auto profile = builder->createOptimizationProfile();
        if (!profile) {
            return false;
        }
        profile->setDimensions(input_name_.c_str(),
                            OptProfileSelector::kMIN, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
        profile->setDimensions(input_name_.c_str(),
                            OptProfileSelector::kOPT, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
        profile->setDimensions(input_name_.c_str(),
                            OptProfileSelector::kMAX, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
        config->addOptimizationProfile(profile);
    }

    auto profile_stream = makeCudaStream();
    if (!profile_stream) {
        return false;
    }
    config->setProfileStream(*profile_stream);
    TensorRTUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }

    runtime_ = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(gLogger.getTRTLogger()));
    if (!runtime_) {
        return false;
    }

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine_) {
        return false;
    }

    SaveEngine();
    return true;
}

bool TensorRTInference::ConstructNetwork(
        TensorRTUniquePtr<nvinfer1::IBuilder>& builder,
        TensorRTUniquePtr<nvinfer1::INetworkDefinition>& network,
        TensorRTUniquePtr<nvinfer1::IBuilderConfig>& config,
        TensorRTUniquePtr<nvonnxparser::IParser>& parser) const {
    
    auto parsed = parser->parseFromFile(onnx_file_.c_str(),
                                        static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed) {
        return false;
    }
    config->setFlag(BuilderFlag::kFP16);
    enableDLA(builder.get(), config.get(), use_dla_);
    return true;
}


bool TensorRTInference::Infer(
               const cv::Mat& image,
               torch::Tensor& line_maps, 
               torch::Tensor& point_maps,
               torch::Tensor& desc_maps) {
    if (!context_) {
        context_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!context_) {
            return false;
        }
    }

    assert(engine_->getNbBindings() == 4);

    const int input_index = engine_->getBindingIndex(input_name_.c_str());

    context_->setBindingDimensions(input_index, Dims4(1, image.channels(), image.rows, image.cols));

    // Create RAII buffer manager object
    BufferManager buffers(engine_, 0, context_.get());

    if (!ProcessInput(buffers, image)) {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context_->executeV2(buffers.getDeviceBindings().data());
    if (!status) {
        return false;
    }
    
    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    if (!ProcessOutput(buffers, image.rows, image.cols, 
                line_maps, point_maps, desc_maps)) {
        return false;
    }

    return true;

}


bool TensorRTInference::ProcessInput(
        const BufferManager& buffers, 
        const cv::Mat& image) const {
        
    cv::Mat input_image;
    image.convertTo(input_image, CV_32F, 1.0/127.5, -1.0);
    cv::Mat rgb[3];
    cv::split(input_image, rgb);
    float* host_databuffer = static_cast<float*>(buffers.getHostBuffer(input_name_));
    memcpy(host_databuffer, rgb[0].data, image.rows * image.cols * sizeof(float));
    memcpy(host_databuffer + image.rows * image.cols, rgb[1].data, image.rows * image.cols * sizeof(float));
    memcpy(host_databuffer + image.rows * image.cols * 2, rgb[2].data, image.rows * image.cols * sizeof(float));
    return true;
}


bool TensorRTInference::ProcessOutput(
        const BufferManager& buffers,
        const int input_height,
        const int input_width,
        torch::Tensor& line_maps, 
        torch::Tensor& point_maps,
        torch::Tensor& desc_maps) const {
    auto ops = torch::TensorOptions().dtype(torch::kFloat);
    float* line_maps_buffer = static_cast<float*>(buffers.getHostBuffer("line_pred"));
    float* point_maps_buffer = static_cast<float*>(buffers.getHostBuffer("points_pred"));
    float* desc_maps_buffer = static_cast<float*>(buffers.getHostBuffer("desc_pred"));

    line_maps = torch::from_blob(line_maps_buffer, {1, 5, input_height / 2, input_width / 2}, ops).clone();
    point_maps = torch::from_blob(point_maps_buffer, {1, 3, input_height / 8, input_width / 8}, ops).clone();
    desc_maps = torch::from_blob(desc_maps_buffer, {1, 128, input_height / 4, input_width / 4}, ops).clone();


    return true;
}


void TensorRTInference::SaveEngine() const {
    std::string engine_file_path = onnx_file_.substr(0, onnx_file_.find_last_of('.')) + ".engine";
    if (engine_ != nullptr) {
        TensorRTUniquePtr<nvinfer1::IHostMemory> serialized_engine(engine_->serialize());
        std::ofstream file(engine_file_path, std::ios::binary);
        if (!file) return;
        file.write(static_cast<char*>(serialized_engine->data()), serialized_engine->size());
    }
}

bool TensorRTInference::LoadEngine() {
    std::string engine_file_path = onnx_file_.substr(0, onnx_file_.find_last_of('.')) + ".engine";
    std::ifstream engine_file(engine_file_path, std::ios::binary);

    if (engine_file.is_open()) {
        engine_file.seekg(0, engine_file.end);
        long int fsize = engine_file.tellg();
        engine_file.seekg(0, engine_file.beg);

        std::vector<char> engine_data(fsize);
        engine_file.read(engine_data.data(), fsize);

        runtime_ = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(gLogger.getTRTLogger()));

        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(engine_data.data(), fsize));
        engine_file.close();
        return true;
    }

    return false;
}



} // namespace likl

#endif


