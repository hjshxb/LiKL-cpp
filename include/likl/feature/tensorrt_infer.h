#pragma once

#ifdef WITH_TENSORRT

#include <memory>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <opencv2/core.hpp>
#include <torch/torch.h>

#include <likl/third_party/tensorrtbuffer/buffers.h>

namespace likl {

class TensorRTInference {
public:
    TensorRTInference(const std::string& onnx_file, const std::string& input_name = "input", const int use_dla = -1)
        : onnx_file_(onnx_file), input_name_(input_name), use_dla_(use_dla),
          engine_(nullptr), context_(nullptr) {}

    // Build the netword engine
    bool Build();

    // Runs the TensorRt inference engine
    bool Infer(const cv::Mat& image,
               torch::Tensor& line_maps, 
               torch::Tensor& point_maps,
               torch::Tensor& desc_maps);

    bool ProcessInput(const tensorrt_buffer::BufferManager& buffers, const cv::Mat& image) const;

    bool ProcessOutput(const tensorrt_buffer::BufferManager& buffers,
                       const int input_height,
                       const int input_width,
                       torch::Tensor& line_maps, 
                       torch::Tensor& point_maps,
                       torch::Tensor& desc_maps) const;
    

private:
    // Uses a ONNX parser to create the Network
    bool ConstructNetwork(tensorrt_common::TensorRTUniquePtr<nvinfer1::IBuilder>& builder,
                        tensorrt_common::TensorRTUniquePtr<nvinfer1::INetworkDefinition>& network,
                        tensorrt_common::TensorRTUniquePtr<nvinfer1::IBuilderConfig>& config,
                        tensorrt_common::TensorRTUniquePtr<nvonnxparser::IParser>& parser) const;
    
    void SaveEngine() const;

    bool LoadEngine();

private:
    std::string onnx_file_;
    std::string input_name_;
    int use_dla_;

    std::shared_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;
};

} // namespace likl

#endif