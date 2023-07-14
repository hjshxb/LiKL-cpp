#pragma once

#include <chrono>

namespace likl {
namespace utils {

class Timer {
public:
    static std::chrono::high_resolution_clock::time_point tic() {
        return std::chrono::high_resolution_clock::now();
    }

    template<typename T = std::chrono::milliseconds>
    static T toc(const std::chrono::high_resolution_clock::time_point& start) {
        return std::chrono::duration_cast<T>(
            std::chrono::high_resolution_clock::now() - start
        );
    }

    
};


} // namespace utils
} // namespace likl