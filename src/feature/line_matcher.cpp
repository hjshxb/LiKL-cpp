/**
 * Refrence from https://github.com/cvg/SOLD2 
*/

#include <glog/logging.h>
#include <torch/torch.h>

#include "likl/feature/line_matcher.h"


namespace likl {

WunschLineMatcher::WunschLineMatcher(bool cross_check,
                                     const torch::DeviceType& device,
                                     int topk_candidates,
                                     float gap)
    : cross_check_(cross_check),
      device_(device), 
      topk_candidates_(topk_candidates),
      gap_(gap) {}


void WunschLineMatcher::Match(
            const cv::Mat& query_desc,
            const cv::Mat& train_desc,
            const cv::Mat& query_mask,
            const cv::Mat& train_mask,
            std::vector<std::pair<int, float>>& matches) const {
    if (query_desc.empty() || train_desc.empty()) {
        LOG(WARNING) << "Descriptors matrices are empty. ";
        return;
    }

    CHECK_EQ(query_mask.rows, query_desc.rows)
        << "Query mask should have " << query_desc.rows << " rows";
    CHECK_EQ(train_mask.rows, train_desc.rows)
        << "Train mask should have " << train_desc.rows << " rows";
    
    const int D = query_desc.cols / query_mask.cols;
    const int num_samples1 = query_mask.cols;
    const int num_samples2 = train_mask.cols;
    const int num_lines1 = query_mask.rows;
    const int num_lines2 = train_mask.rows;

    torch::NoGradGuard no_grad;
    auto ops = torch::TensorOptions().dtype(torch::kFloat);
    torch::Tensor query_desc_tensor = torch::from_blob(query_desc.data, {num_lines1 * num_samples1, D}, ops);
    torch::Tensor train_desc_tensor = torch::from_blob(train_desc.data, {num_lines2 * num_samples2, D}, ops);
    query_desc_tensor = query_desc_tensor.to(device_);
    train_desc_tensor = train_desc_tensor.to(device_);
    ops = torch::TensorOptions().dtype(torch::kBool);
    torch::Tensor query_mask_tensor = torch::from_blob(query_mask.data, {num_lines1 * num_samples1}, ops);
    torch::Tensor train_mask_tensor = torch::from_blob(train_mask.data, {num_lines2 * num_samples2}, ops);
    query_mask_tensor = query_mask_tensor.to(device_);
    train_mask_tensor = train_mask_tensor.to(device_);

    // Cosine Distance (num_lines1 * num_samples1, num_lines2 * num_samples2)
    torch::Tensor scores_tensor = torch::mm(query_desc_tensor, train_desc_tensor.t());
    // Assign a score of -2 for unvalid points
    scores_tensor.index_put_({query_mask_tensor}, -2);
    scores_tensor.index_put_({"...", train_mask_tensor}, -2);
    // (num_lines1, num_lines2, num_samples1, num_samples2)
    scores_tensor = scores_tensor.reshape({num_lines1, num_samples1, num_lines2, num_samples2});
    scores_tensor = scores_tensor.permute({0, 2, 1, 3});

    MatchLinesImpl(scores_tensor, matches);
    if (cross_check_) {
        std::vector<std::pair<int, float>> matches2to1;
        MatchLinesImpl(scores_tensor.permute({1, 0, 3, 2}), matches2to1);

        for (size_t i = 0; i < matches.size(); ++i) {
            int idx = matches[i].first;
            if (idx != -1 && matches2to1.at(idx).first != static_cast<int>(i)) {
                matches[i].first = -1;
            }
        }
    }
}


void WunschLineMatcher::MatchLinesImpl(
        const torch::Tensor& scores_tensor,
        std::vector<std::pair<int, float>>& matches) const {
    const int num_lines1 = scores_tensor.size(0);
    // Pre-filter the pairs and keep the topk best candidate lines
    torch::Tensor line_scores1 = std::get<0>(scores_tensor.max(3));
    torch::Tensor valid_mask1 = (line_scores1 != -2);
    line_scores1 = (line_scores1 * valid_mask1).sum(2);
    line_scores1.div_(valid_mask1.sum(2));

    torch::Tensor line_scores2 = std::get<0>(scores_tensor.max(2));
    torch::Tensor valid_mask2 = (line_scores2 != -2);
    line_scores2 = (line_scores2 * valid_mask2).sum(2);
    line_scores2.div_(valid_mask2.sum(2));
    
    torch::Tensor line_scores = (line_scores1 + line_scores2) / 2;
    using namespace torch::indexing;
    torch::Tensor topk_lines_index = 
        torch::argsort(line_scores, 1, true)
            .index({Slice(), Slice(None, topk_candidates_)});
    torch::Tensor topk_scores = torch::take_along_dim(
        scores_tensor, topk_lines_index.view({num_lines1, -1, 1, 1}), 1);
    
    // NeedlemanWunsch(topk_scores, topk_lines_index, matches);
    BatchNeedlemanWunsch(topk_scores, topk_lines_index, matches);
}


void WunschLineMatcher::NeedlemanWunsch(
            const torch::Tensor& topk_scores,
            const torch::Tensor& topk_lines_index, 
            std::vector<std::pair<int, float>>& matches) const {
    int num_lines1 = topk_scores.size(0);
    matches.resize(num_lines1, std::make_pair<int, float>(-1, 0));

    int topk = topk_scores.size(1);
    torch::Tensor topk_lines_index_cpu = topk_lines_index.cpu();
    auto topk_index_data = topk_lines_index_cpu.accessor<long, 2>();

    cv::Size grid_size(topk_scores.size(3), topk_scores.size(2));
    using namespace torch::indexing;
    for (int i = 0; i < num_lines1; ++i) {
        int match_idx = -1;
        float max_score = -2 * topk_scores.size(3);
        for (int j = 0; j < topk; ++j) {
            float* cur_scores_data = topk_scores.index({i, j, Slice(), Slice()})
                                     .cpu()
                                     .contiguous()
                                     .data_ptr<float>();
            cv::Mat scores_mat(grid_size, CV_32FC1, cur_scores_data);
            float score = DescriptorScore(scores_mat, gap_, false);
            if (score > max_score) {
                max_score = score;
                match_idx = j;
            }
            // Consider the reversed line segments as well 
            score = DescriptorScore(scores_mat, gap_, true);
            if (score > max_score) {
                max_score = score;
                match_idx = j + topk;
            }
        }

        matches[i].first = topk_index_data[i][match_idx % (topk)];
        matches[i].second = max_score;
    }

}


void WunschLineMatcher::BatchNeedlemanWunsch(
            const torch::Tensor& topk_scores,
            const torch::Tensor& topk_lines_index,
            std::vector<std::pair<int, float>>& matches) const {
    const int num_lines1 = topk_scores.size(0);
    const int topk = topk_scores.size(1);
    matches.resize(num_lines1, std::make_pair<int, float>(-1, 0));

    const int rows = topk_scores.size(2);
    const int cols = topk_scores.size(3);
    // Consider the reversed line segments as well
    torch::Tensor top2k_scores = torch::cat({topk_scores, topk_scores.flip({3})}, 1);
    top2k_scores = top2k_scores.reshape({num_lines1 * topk * 2, rows, cols});
    top2k_scores.sub_(gap_);

    auto ops = torch::TensorOptions().dtype(torch::kFloat).device(top2k_scores.device());
    torch::Tensor nw_grid = torch::zeros({num_lines1 * topk * 2, rows + 1, cols + 1}, ops);

    using namespace torch::indexing;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            torch::Tensor tmp_max = torch::maximum(
                nw_grid.index({Slice(), i + 1, j}),
                nw_grid.index({Slice(), i , j + 1})
            );
            torch::Tensor max_value = torch::maximum(
                tmp_max,
                nw_grid.index({Slice(), i, j}) + top2k_scores.index({Slice(), i, j})
            );
            nw_grid.index_put_({Slice(), i + 1, j + 1}, max_value);
        }
    }

    torch::Tensor final_scores = nw_grid.index({Slice(), rows, cols})
                                    .reshape({num_lines1, 2 * topk});
    torch::Tensor max_indices = torch::argmax(final_scores, 1).fmod_(topk);
    
    max_indices = max_indices.cpu();
    final_scores = final_scores.cpu();
    auto max_indices_data = max_indices.accessor<long, 1>();
    auto final_scores_data = final_scores.accessor<float, 2>();
    torch::Tensor topk_lines_index_cpu = topk_lines_index.cpu();
    auto topk_index_data = topk_lines_index_cpu.accessor<long, 2>();

    for (int i = 0; i < num_lines1; ++i) {
        matches[i].first = topk_index_data[i][max_indices_data[i]];
        matches[i].second = final_scores_data[i][max_indices_data[i]];
    }
}

float WunschLineMatcher::DescriptorScore(
            const cv::Mat& scores_mat,
            float gap,
            bool reverse) {
    CV_Assert(scores_mat.depth() == CV_32F);
    const int rows = scores_mat.rows;
    const int cols = scores_mat.cols;
    cv::Mat_<float> nw_grid(rows + 1, cols + 1, 0.0f);

    cv::Mat nw_scores = scores_mat - gap;
    for (int i = 0; i < rows; ++i) {
        float* s_ptr = nw_scores.ptr<float>(i);
        for (int j = 0; j < cols; ++j) {
            int visit_j = reverse ? cols - j - 1 : j;
            nw_grid(i + 1, j + 1) = std::max(
                {nw_grid(i + 1, j),
                 nw_grid(i, j + 1),
                 nw_grid(i, j) + s_ptr[visit_j]}
            );
        }
    }
    return nw_grid(rows, cols);
}

} // namespace likl