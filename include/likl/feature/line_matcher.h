#pragma once

#include <opencv2/core.hpp>
#include <torch/data.h>


namespace likl {

class WunschLineMatcher {
public:
    WunschLineMatcher() = delete;

    WunschLineMatcher(bool cross_check,
                      const torch::DeviceType& device = torch::kCPU,
                      int topk_candidates = 10,
                      float gap = 0.1);
    
    void Match(const cv::Mat& query_desc,
               const cv::Mat& train_desc,
               const cv::Mat& query_mask,
               const cv::Mat& train_mask,
               std::vector<std::pair<int, float>>& matches) const;
    
    static void ToDmatch(const std::vector<std::pair<int, float>>& matches, 
                         std::vector<cv::DMatch>& Dmatches);
    
    /**
     * @brief:  The implementation of the Needleman-Wunsch algorithm 
     *          to get descriptor score
     * @param scores_mat: (N，M) cv::Mat
     * @param reverse: Reverse order visit scores mat along columns
    */
    static float DescriptorScore(
                    const cv::Mat& scores_mat,
                    float gap,
                    bool reverse);

private:
    void MatchLinesImpl(const torch::Tensor& scores_tensor,
                        std::vector<std::pair<int, float>>& matches) const;

    void NeedlemanWunsch(
                const torch::Tensor& topk_scores,
                const torch::Tensor& topk_lines_index, 
                std::vector<std::pair<int, float>>& matches) const;

    void BatchNeedlemanWunsch(
                const torch::Tensor& topk_scores,
                const torch::Tensor& topk_lines_index,
                std::vector<std::pair<int, float>>& matches) const;

public:
    bool cross_check_;
    torch::Device device_;
    int topk_candidates_;
    float gap_;

};



} // namespace likl
