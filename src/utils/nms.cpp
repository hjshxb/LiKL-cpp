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
        case AnmsAlgorithmType::Binning: {
            nms_kpts = binning(sort_kpts, num_retPoints, cols, rows, 50, 12);
            break;
        }
        default:
            LOG(WARNING) << "Unknown nms algorithm...";
            break;
    }
    return nms_kpts;

}

std::vector<cv::KeyPoint> AdaptiveNMS::binning(
    const std::vector<cv::KeyPoint>& keyPoints,
    const int& numKptsToRetain,
    const int& imgCols,
    const int& imgRows,
    const int& nr_horizontal_bins,
    const int& nr_vertical_bins) const{
  if (static_cast<size_t>(numKptsToRetain) > keyPoints.size()) {
    return keyPoints;
  }

  float binRowSize = float(imgRows) / float(nr_vertical_bins);
  float binColSize = float(imgCols) / float(nr_horizontal_bins);

  // Note: features should be already sorted by score at this point from detect

  // 0. count the number of valid bins (as specified by the user in the yaml
  // TODO： set binning_mask using param
  Eigen::MatrixXd binning_mask = Eigen::MatrixXd::Ones(nr_vertical_bins, nr_horizontal_bins);
  float nrActiveBins = binning_mask.sum();  // sum of 1's in binary mask

  // 1. compute how many features we want to retain in each bin
  // numRetPointsPerBin
  const int numRetPointsPerBin =
      std::round(float(numKptsToRetain) / float(nrActiveBins));

  // 2. assign keypoints to bins and retain top numRetPointsPerBin for each bin
  std::vector<cv::KeyPoint> binnedKpts;  // binned keypoints we want to output
  Eigen::MatrixXd nrKptsInBin = Eigen::MatrixXd::Zero(
      nr_vertical_bins,
      nr_horizontal_bins);  // store number of kpts for each bin
  for (size_t i = 0; i < keyPoints.size(); i++) {
    const size_t binRowInd =
        static_cast<size_t>(keyPoints[i].pt.y / binRowSize);
    const size_t binColInd =
        static_cast<size_t>(keyPoints[i].pt.x / binColSize);
    // if bin is active and needs more keypoints
    if (binning_mask(binRowInd, binColInd) == 1 &&
        nrKptsInBin(binRowInd, binColInd) <
            numRetPointsPerBin) {  // if we need more kpts in that bin
      binnedKpts.push_back(keyPoints[i]);
      nrKptsInBin(binRowInd, binColInd) += 1;
    }
  }
  return binnedKpts;
}


} // namespace utils
} // namespace likl