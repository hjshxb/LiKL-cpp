#pragma once 

#include <Eigen/Core>
/**
 * For more details, please refer to https://github.com/BAILOOL/ANMS-Codes/tree/master
*/
#include "likl/common_data_type.h"
#include "likl/third_party/anms/anms.h"

namespace likl {
namespace utils {

class AdaptiveNMS {
public:
    AdaptiveNMS() = delete;
    AdaptiveNMS(const AnmsAlgorithmType& anms_algorithm_type);

    std::vector<cv::KeyPoint> run(
        const std::vector<cv::KeyPoint>& keyPoints,
        int numRetPoints,
        float tolerance,
        int cols,
        int rows) const;
    
    std::vector<cv::KeyPoint> binning(
        const std::vector<cv::KeyPoint>& keyPoints,
        const int& numKptsToRetain,
        const int& imgCols,
        const int& imgRows,
        const int& nr_horizontal_bins,
        const int& nr_vertical_bins) const;

private:
    AnmsAlgorithmType anms_algorithm_type_;

};

} // namespace utils
} // namespace likl