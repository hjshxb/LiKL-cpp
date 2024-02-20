#pragma once

namespace likl {

/**
 * @brief: Types of supported camera model
 */
enum class AnmsAlgorithmType {
    TopN = 0,
    BrownANMS = 1,
    Sdc = 2,
    KdTree = 3,
    RangeTree = 4,
    Ssc = 5,
    Binning = 6
};

}  // namespace likl