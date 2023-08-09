
#include "likl/param/feature_param.h"

namespace likl {

const AnmsAlgorithmType String2AnmsAlgorithmType(const std::string& nms_type) {
    std::string lower_nms_type = nms_type;
    std::transform(lower_nms_type.begin(),
                   lower_nms_type.end(),
                   lower_nms_type.begin(),
                   ::tolower);
    if (lower_nms_type == "topn") {
        return AnmsAlgorithmType::TopN;
    } else if (lower_nms_type == "brownanms") {
        return AnmsAlgorithmType::BrownANMS;
    } else if (lower_nms_type == "sdc") {
        return AnmsAlgorithmType::Sdc;
    } else if (lower_nms_type == "kdtree") {
        return AnmsAlgorithmType::KdTree;
    } else if (lower_nms_type == "rangetree") {
        return AnmsAlgorithmType::RangeTree;
    } else if (lower_nms_type == "ssc") {
        return AnmsAlgorithmType::Ssc;
    } else {
        LOG(FATAL) << "Invalid nms type, Valid nms types are"
                   << "  topn sdc kdtree rangetree ssc";
    }
}

void FeatureParams::ParseParam(const cv::FileNode& file_node) {
    GetNodeValue(file_node, "max_keypoints", num_keypoints_);
    GetNodeValue(file_node, "keypoints_threshold", keypoints_threshold_);
    std::string nms_type;
    GetNodeValue(file_node, "nms_type", nms_type);
    nms_type_ = String2AnmsAlgorithmType(nms_type);

    GetNodeValue(file_node, "model_path", model_path_);
    GetNodeValue(file_node, "grid_size", grid_size_);
    GetNodeValue(file_node, "cross_ratio", cross_ratio_);
    GetNodeValue(file_node, "tensor_width", tensor_width_);
    GetNodeValue(file_node, "tensor_height", tensor_height_);
    GetNodeValue(file_node, "line_threshold", line_threshold_);
    GetNodeValue(file_node, "min_line_length", min_line_length_);
    GetNodeValue(file_node, "max_num_sample", max_num_sample_),
    GetNodeValue(file_node, "min_sample_dist", min_sample_dist_);

    // It needs to be ensured to be a multiple of 32
    if (tensor_height_ % 32 != 0) {
        tensor_height_ = ceil(tensor_height_ / 32.0f) * 32;
    }

    if (tensor_width_ % 32 != 0) {
        tensor_width_ = ceil(tensor_width_ / 32.0f) * 32;
    }

}


void FeatureParams::Print() const {
    std::stringstream ss;
    Params::Print(ss, 
                  "num_keypoints", num_keypoints_,
                  "keypoints_threshold", keypoints_threshold_,
                  "model_path", model_path_,
                  "grid_size", grid_size_,
                  "cross_ratio", cross_ratio_,
                  "tensor_width", tensor_width_,
                  "tensor_height", tensor_height_,
                  "line_threshold", line_threshold_,
                  "max_num_sample", max_num_sample_,
                  "min_sample_dist", min_sample_dist_);
    LOG(INFO) << ss.str();
}

} // namespace likl