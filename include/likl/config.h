#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include "likl/param/param.h"

namespace likl {


template<typename ParamsType>
class Config {
    
public:
    Config(const std::string& config_path, const std::string& node_name);
    Config(const cv::FileNode& file_node);

    virtual ~Config() {
        if (file_storage_.isOpened()) {
            file_storage_.release();
        }
    }

    void Print() const;


    ParamsType params_;

private:
    const std::string config_path_;
    cv::FileStorage file_storage_;

};


template<typename ParamsType>
Config<ParamsType>::Config(const std::string& config_path, const std::string& node_name)
    : config_path_(config_path), file_storage_(config_path, cv::FileStorage::READ) {
        CHECK(file_storage_.isOpened()) << "Can not open :" << config_path;
        LOG(INFO) << "Load config params from: " << config_path;
        LOG(INFO) << "Load params";
        params_.ParseParam(file_storage_[node_name]);
    }

template<typename ParamsType>
Config<ParamsType>::Config(const cv::FileNode& file_node) {
    params_.ParseParam(file_node);
}


template<typename ParamsType>
void Config<ParamsType>::Print() const {
    LOG(INFO) << "Config file path: "
              << config_path_;
    params_.Print();
}




} // namespace likl
