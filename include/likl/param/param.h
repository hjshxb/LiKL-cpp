#pragma once 

// Reference: https://github.com/MIT-SPARK/Kimera-VIO/blob/master/src/pipeline/PipelineParams.cpp

#include <iostream>
#include <sstream>
#include <glog/logging.h>
#include <opencv2/core.hpp>

namespace likl {

class Params {
/**
 * @brief base class of parameters
 */
public:
    explicit Params(const std::string& name) : params_name_(name) {}
    virtual ~Params() = default;

    virtual void ParseParam(const cv::FileNode& file_node) = 0;
    
    virtual void Print() const = 0;

    
    // FileNode Parser
    template<typename T>
    void GetNodeValue(const cv::FileNode& fnode, const std::string& id, T& value) {
        const cv::FileNode& node = fnode[id];
        CHECK_NE(node.type(), cv::FileNode::NONE) 
            << "Missing param: " << id.c_str();
        value = static_cast<T>(node);
    }


    template <typename ParamsType>
    ParamsType* as() noexcept {
        return dynamic_cast<ParamsType*>(this);
    }

    template <typename ParamsType>
    const ParamsType* as() const noexcept {
        return dynamic_cast<const ParamsType*>(this);
    }

protected:
    std::string params_name_;
    static constexpr size_t kNameWidth = 40;
    static constexpr size_t kValueWidth = 20;
    static constexpr size_t kTotalWidth = kNameWidth + kValueWidth;

    template<typename... Args>
    void Print(std::stringstream& out, Args... to_print) const {
        out.str("");

        // Add title.
        out.width(kTotalWidth);
        size_t center =
            (kTotalWidth - params_name_.size() - 2u) / 2u;  // -2u for ' ' chars
        out << '\n'
            << std::string(center, '*').c_str() << ' ' << params_name_.c_str() << ' '
            << std::string(center, '*').c_str() << '\n';

        // Add columns' headers.
        out.width(kNameWidth);  // Remove hardcoded, need to pre-calc width.
        out.setf(std::ios::left, std::ios::adjustfield);
        out << "Name";
        out.setf(std::ios::right, std::ios::adjustfield);
        out.width(kValueWidth);
        out << "Value\n";

        PrintLine(out, '.');

        PrintImpl(out, to_print...);

        // Add horizontal separator
        PrintLine(out);
    }


private:

    static inline void PrintLine(std::stringstream& out, char f = ' ') {
        out.width(kTotalWidth);
        out.fill('-');
        out << "\n";
        out.fill(f);
    }


    template<typename TName, typename TValue>
    void PrintImpl(std::stringstream& out, TName name, TValue value) const {
        out.width(kNameWidth - 1);
        out.setf(std::ios::left, std::ios::adjustfield);
        out << name;
        out.setf(std::ios::right, std::ios::adjustfield);
        out.width(kValueWidth);
        out << value << '\n';
    }

    template<typename TName, typename TValue, typename... Args>
    void PrintImpl(std::stringstream& out, 
                   TName name,
                   TValue value,
                   Args... next) const {
        out.width(kNameWidth - 1);
        out.setf(std::ios::left, std::ios::adjustfield);
        out << name;
        out.width(kValueWidth);
        out.setf(std::ios::right, std::ios::adjustfield);
        out << value << "\n";
        PrintImpl(out, next...);
    }

};

} // namespace likl