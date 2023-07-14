#pragma once

#include <torch/script.h>
#include <eigen3/Eigen/Dense>

#include "likl/param/feature_param.h"

namespace likl {

class LiKL {
public:
    LiKL() = delete;
    LiKL(const FeatureParams& params, const torch::DeviceType& device);

    /**
     * @brief: Detect point and line features
     * @param: detect_mask: it specifies the region in which the point are detected
    */
    void Detect(const cv::Mat& img, 
                std::vector<cv::KeyPoint>& keypoints, 
                cv::Mat& kps_descriptors,
                std::vector<cv::Vec4f>& lines,
                cv::Mat& line_descriptors,
                cv::Mat& line_descr_mask,
                cv::InputArray detect_mask = cv::noArray());
    
    void Detect(const std::vector<cv::Mat>& vec_img,
                std::vector<std::vector<cv::KeyPoint>>& vec_keypoints,
                std::vector<cv::Mat>& vec_kps_descriptors,
                std::vector<std::vector<cv::Vec4f>>& vec_lines,
                std::vector<cv::Mat>& vec_line_descriptors,
                std::vector<cv::Mat>& vec_line_descr_mask,
                const std::vector<cv::Mat>& vec_detect_mask);

    torch::Tensor SimpleNms(const torch::Tensor& heatmap, 
                            int radius = 1,
                            int stride = 1) const;

    void SamplePointOnLine(const std::vector<cv::Vec4f>& lines,
                           std::vector<cv::Vec2f>& points_on_lines,
                           cv::Mat& mask) const;

    void WarmUp(int iter = 10);

    inline double clamp(double val, double low, double high) const {
      return (val < low) ? low : (high < val) ? high : val;
    }

private:

    void Forward(const std::vector<cv::Mat>& imgs,
                 torch::Tensor& point_maps,
                 torch::Tensor& line_maps,
                 torch::Tensor& desc_maps);

    // postprocessing
    void GetKeypoints(const torch::Tensor& point_maps,
                      const cv::Size& img_size,
                      std::vector<cv::KeyPoint>& keypoints,
                      cv::InputArray detect_mask = cv::noArray()) const;

    void ComputeDescriptors(const torch::Tensor& desc_map, 
                            const std::vector<cv::KeyPoint>& keypoints,
                            const cv::Size& img_size,
                            cv::Mat& descriptors) const;

    void ComputeDescriptors(const torch::Tensor& desc_map,
                            const std::vector<cv::Vec2f>& keypoints,
                            const cv::Size& img_size,
                            cv::Mat& descriptors) const;

    void _ComputeDescriptorsImpl(const torch::Tensor& desc_map,
                                 const torch::Tensor& grid_tensor,
                                 cv::Mat& descriptors) const;

    
    void GetLines(const torch::Tensor& line_maps,
                  const cv::Size& img_size,
                  std::vector<cv::Vec4f>& lines) const;

    torch::Tensor CreateImageGrid(const cv::Size& img_size) const;


private:
    const FeatureParams params_;
    torch::jit::Module module_;
    torch::Device device_;


};

} // namespace likl