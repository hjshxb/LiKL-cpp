#pragma once 

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

private:
    AnmsAlgorithmType anms_algorithm_type_;

};

} // namespace utils
} // namespace likl