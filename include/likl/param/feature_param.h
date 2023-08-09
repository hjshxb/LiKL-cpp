#pragma once
#include "likl/common_data_type.h"
#include "likl/param/param.h"

namespace likl {

const AnmsAlgorithmType String2AnmsAlgorithmType(const std::string& nms_type);

class FeatureParams : public Params {
public:
    FeatureParams()
            : Params("Feature Parameters"),
            num_keypoints_(),
            keypoints_threshold_(),
            nms_type_(AnmsAlgorithmType::RangeTree),
            model_path_(),
            grid_size_(),
            cross_ratio_(),
            tensor_height_(),
            tensor_width_(),
            line_threshold_(),
            min_line_length_(),
            max_num_sample_(),
            min_sample_dist_()
            {}

    virtual ~FeatureParams() = default;

    // parse FileNode to get params
    void ParseParam(const cv::FileNode& file_node) override;

    // Display all params
    void Print() const override;


public:
    int num_keypoints_;
    float keypoints_threshold_;
    AnmsAlgorithmType nms_type_;

    // For lkle
    std::string model_path_;
    int grid_size_;
    float cross_ratio_;
    int tensor_height_;
    int tensor_width_;
    float line_threshold_;
    float min_line_length_;
    float max_num_sample_;
    float min_sample_dist_;

};

} // namespace likl