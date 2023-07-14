#include <algorithm>
#include <glog/logging.h>

#include "likl/utils/nms.h"

namespace likl {
namespace utils {

AdaptiveNMS::AdaptiveNMS(const AnmsAlgorithmType& anms_algorithm_type)
    : anms_algorithm_type_(anms_algorithm_type) {}

std::vector<cv::KeyPoint> AdaptiveNMS::run(
        const std::vector<cv::KeyPoint>& keypoints,
        const int num_retPoints,
        const float tolerance,
        const int cols,
        const int rows) const {
    if (keypoints.empty()) {
        LOG(WARNING) << "no keypoints to nms...";
        return std::vector<cv::KeyPoint>();
    }
    std::vector<cv::KeyPoint> sort_kpts(keypoints);
    // Sorting keypoints by deacreasing order of strength
    std::sort(sort_kpts.begin(), sort_kpts.end(), 
        [](const cv::KeyPoint& kp1, const cv::KeyPoint& kp2) {
            return kp1.response > kp2.response;});
    
    std::vector<cv::KeyPoint> nms_kpts;
    switch (anms_algorithm_type_) {
        case AnmsAlgorithmType::TopN: {
            nms_kpts = anms::TopN(sort_kpts, num_retPoints);
            break;
        }
        case AnmsAlgorithmType::BrownANMS: {
            nms_kpts = anms::BrownANMS(sort_kpts, num_retPoints);
            break;
        }
        case AnmsAlgorithmType::Sdc: {
            nms_kpts = anms::Sdc(sort_kpts, num_retPoints, tolerance, cols, rows);
            break;
        }
        case AnmsAlgorithmType::KdTree: {
            nms_kpts = anms::KdTree(sort_kpts, num_retPoints, tolerance, cols, rows);
            break;
        }
        case AnmsAlgorithmType::RangeTree: {
            nms_kpts = anms::RangeTree(sort_kpts, num_retPoints, tolerance, cols, rows);
            break;
        }
        case AnmsAlgorithmType::Ssc: {
            nms_kpts = anms::Ssc(sort_kpts, num_retPoints, tolerance, cols, rows);
            break;
        }
        default:
            LOG(WARNING) << "Unknown nms algorithm...";
            break;
    }
    return nms_kpts;

}


} // namespace utils
} // namespace likl