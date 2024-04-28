#include <glog/logging.h>
#include <torch/torch.h>
#include <torchvision/vision.h>

#include "likl/utils/nms.h"
#include "likl/utils/timer.h"
#include "likl/feature/likl_extractor.h"

namespace likl {

LiKL::LiKL(const FeatureParams& params, const torch::DeviceType& device)
        : params_(params), device_(device) {

#ifdef WITH_TENSORRT

    //Loading TensorRT engine
    LOG(INFO)<< "Loading TensorRT engine";
    module_ = std::make_shared<TensorRTInference>(params_.model_path_, "input", params_.dla_core_);
    module_->Build();
    
#else

    // Loading the torchscript model
    try {
        module_ = torch::jit::load(params_.model_path_);
    }
    catch (const c10::Error& e) {
        LOG(ERROR) << "Error loading the model\n" 
                   << e.msg();
        std::exit(EXIT_FAILURE);
    }
    module_.to(device);
    module_.eval();

#endif
    
}


void LiKL::Detect(
            const cv::Mat& img, 
            std::vector<cv::KeyPoint>& keypoints, 
            cv::Mat& kps_descriptors,
            std::vector<cv::Vec4f>& lines,
            cv::Mat& line_descriptors,
            cv::Mat& line_descr_mask,
            cv::InputArray detect_mask) {
    CV_Assert(detect_mask.empty() || 
              (detect_mask.type() == CV_8UC1 && detect_mask.sameSize(img)));
    torch::NoGradGuard no_grad;

    std::vector<cv::Mat> vec_img = {img};
    torch::Tensor point_maps;
    torch::Tensor line_maps;
    torch::Tensor desc_maps;
    Forward(vec_img, point_maps, line_maps, desc_maps);

    cv::Size img_size(img.cols, img.rows);

    // Keypoints postprocessing
    GetKeypoints(point_maps, img_size, keypoints, detect_mask);

    // Compute keypoint descriptors
    ComputeDescriptors(desc_maps, keypoints, img_size, kps_descriptors);

    // Lines postprocessing
    GetLines(line_maps, img_size, lines);

    std::vector<cv::Vec2f> points_on_lines;
    SamplePointOnLine(lines, points_on_lines, line_descr_mask);

    // Compute descriptors of sampling points;
    cv::Mat tmp_line_descr;
    ComputeDescriptors(desc_maps, points_on_lines, img_size, tmp_line_descr);
    line_descriptors = cv::Mat(cv::Size(tmp_line_descr.cols * params_.max_num_sample_,
                                        tmp_line_descr.rows / params_.max_num_sample_),
                               CV_32FC1,
                               tmp_line_descr.data).clone();
}


void LiKL::Detect(
            const std::vector<cv::Mat>& vec_img,
            std::vector<std::vector<cv::KeyPoint>>& vec_keypoints,
            std::vector<cv::Mat>& vec_kps_descriptors,
            std::vector<std::vector<cv::Vec4f>>& vec_lines,
            std::vector<cv::Mat>& vec_line_descriptors,
            std::vector<cv::Mat>& vec_line_descr_mask,
            const std::vector<cv::Mat>& vec_detect_mask) {
    torch::NoGradGuard no_grad;

    torch::Tensor point_maps;
    torch::Tensor line_maps;
    torch::Tensor desc_maps;
    Forward(vec_img, point_maps, line_maps, desc_maps);

    for (size_t i = 0; i < vec_img.size(); ++i) {
        cv::InputArray detect_mask = vec_detect_mask.at(i);

        CV_Assert(detect_mask.empty() || 
                    (detect_mask.type() == CV_8UC1 && detect_mask.sameSize(vec_img[i])));

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat kps_descriptors;
        std::vector<cv::Vec4f> lines;
        cv::Mat line_descriptors;
        cv::Mat line_descr_mask;

        cv::Size img_size(vec_img[i].cols, vec_img[i].rows);
        GetKeypoints(point_maps.slice(0, i, i+1), img_size, keypoints, detect_mask);
        ComputeDescriptors(desc_maps.slice(0, i, i+1), keypoints, img_size, kps_descriptors);
        GetLines(line_maps.slice(0, i, i+1), img_size, lines);

        std::vector<cv::Vec2f> points_on_lines;
        SamplePointOnLine(lines, points_on_lines, line_descr_mask);

        cv::Mat tmp_line_descr;
        ComputeDescriptors(desc_maps.slice(0, i, i+1), points_on_lines, img_size, tmp_line_descr);
        line_descriptors = cv::Mat(cv::Size(tmp_line_descr.cols * params_.max_num_sample_,
                                            tmp_line_descr.rows / params_.max_num_sample_),
                                   CV_32FC1,
                                   tmp_line_descr.data).clone();
        
        vec_keypoints.push_back(std::move(keypoints));
        vec_kps_descriptors.push_back(std::move(kps_descriptors));
        vec_lines.push_back(std::move(lines));
        vec_line_descriptors.push_back(std::move(line_descriptors));
        vec_line_descr_mask.push_back(std::move(line_descr_mask));
    }
}

torch::Tensor LiKL::SimpleNms(
        const torch::Tensor& heatmap, 
        int radius, int stride) const {
    namespace F = torch::nn::functional;
    auto ops = F::MaxPool2dFuncOptions(radius * 2 + 1)
                            .padding(radius)
                            .stride(stride);
    torch::Tensor heatmap_max = F::max_pool2d(heatmap, ops);
    torch::Tensor keep_mask = (heatmap == heatmap_max);
    return heatmap * keep_mask;
}


void LiKL::SamplePointOnLine(
        const std::vector<cv::Vec4f>& lines,
        std::vector<cv::Vec2f>& points_on_lines,
        cv::Mat& mask) const {
    const Eigen::ArrayX4f lines_array = 
            Eigen::Map<const Eigen::ArrayX4f,
                Eigen::Unaligned,
                Eigen::Stride<1, 4>>(&lines[0][0], lines.size(), 4);
    Eigen::ArrayXf lines_length = 
        (lines_array.col(2) - lines_array.col(0)).square() +
        (lines_array.col(3) - lines_array.col(1)).square();
    lines_length = lines_length.sqrt();

    points_on_lines.clear();
    points_on_lines.reserve(lines.size() * params_.max_num_sample_);
    mask.create(lines.size(), params_.max_num_sample_, CV_8UC1);

    for (size_t i = 0; i < lines.size(); ++i) {
        int num_sample = 
            clamp(lines_length(i) / params_.min_sample_dist_,
                  2,
                  params_.max_num_sample_);
        Eigen::ArrayXf line_points_x = Eigen::ArrayXf::LinSpaced(
                num_sample, lines_array(i, 0), lines_array(i, 2));
        Eigen::ArrayXf line_points_y = Eigen::ArrayXf::LinSpaced(
                num_sample, lines_array(i, 1), lines_array(i, 3));
        
        uchar* cur_mask_ptr = mask.ptr<uchar>(i);
        for (int j = 0; j < num_sample; ++j) {
            points_on_lines.emplace_back(line_points_x(j), line_points_y(j));
            cur_mask_ptr[j] = 0;
        }

        // Padding if not enough sampling points
        for (int k = num_sample; k < params_.max_num_sample_; ++k) {
            points_on_lines.emplace_back(0, 0);
            cur_mask_ptr[k] = 1;
        }
    }
}


void LiKL::WarmUp(int iter) {
    cv::Mat input_img(cv::Size(params_.tensor_width_, params_.tensor_height_), 
                      CV_8UC3);
    cv::randu(input_img, cv::Scalar(0, 0, 0), cv::Scalar(256, 256, 256));
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat kps_descriptors;
    std::vector<cv::Vec4f> lines;
    cv::Mat line_descriptors;
    cv::Mat line_descr_mask;
    while(iter--) {
        Detect(input_img, keypoints, kps_descriptors, 
            lines, line_descriptors, line_descr_mask);
    }
}


void LiKL::Forward(
        const std::vector<cv::Mat>& imgs,
        torch::Tensor& point_maps,
        torch::Tensor& line_maps,
        torch::Tensor& desc_maps) {
    
    torch::NoGradGuard no_grad;
    // Get params
    int tensor_width = params_.tensor_width_;
    int tensor_height = params_.tensor_height_;

    // Process img
    std::vector<cv::Mat> input_imgs(imgs.size());
    for (size_t i = 0; i < imgs.size(); ++i) {
        if (tensor_width != imgs[i].cols || tensor_height != imgs[i].rows) {
            cv::resize(imgs[i], input_imgs[i], cv::Size(tensor_width, tensor_height));
        } else {
            imgs[i].copyTo(input_imgs[i]);
        }
        
        if (input_imgs[i].channels() == 1) {
            cv::cvtColor(input_imgs[i], input_imgs[i], cv::COLOR_GRAY2RGB);
        }
    }

#ifdef WITH_TENSORRT
    std::vector<torch::Tensor> vec_line_map(input_imgs.size());
    std::vector<torch::Tensor> vec_point_map(input_imgs.size());
    std::vector<torch::Tensor> vec_desc_map(input_imgs.size());

    // Only `batch_size = 1` is supported.
    for (size_t i = 0; i < input_imgs.size(); ++i) {
        module_->Infer(input_imgs[i], vec_line_map[i], vec_point_map[i], vec_desc_map[i]);
    }

    line_maps = torch::cat(vec_line_map, 0);
    point_maps = torch::cat(vec_point_map, 0);
    desc_maps = torch::cat(vec_desc_map, 0);

#else
    cv::Mat batch_img;
    cv::vconcat(input_imgs, batch_img);
    torch::Tensor tensor_img = torch::from_blob(batch_img.data, 
        {static_cast<int>(input_imgs.size()), input_imgs[0].rows, 
         input_imgs[0].cols, input_imgs[0].channels()}, 
        torch::kByte).to(device_);
    // BHWC ==> BCHW
    tensor_img = tensor_img.permute({0, 3, 1, 2}).to(torch::kFloat) / 127.5 - 1;

    auto output = module_.forward({tensor_img}).toGenericDict();
    line_maps = output.at("line_pred").toTensor();
    point_maps = output.at("points_pred").toTensor();
    desc_maps = output.at("desc_pred").toTensor();

#endif

}

void LiKL::GetKeypoints(
        const torch::Tensor& point_maps,
        const cv::Size& img_size,
        std::vector<cv::KeyPoint>& keypoints,
        cv::InputArray detect_mask) const {
    
    torch::Tensor scores = torch::sigmoid(point_maps.select(1, 0)).squeeze(0);
    torch::Tensor center_shift = torch::tanh(point_maps.slice(1, 1, 3)).squeeze(0);
    float step = (params_.grid_size_ - 1) / 2.0f;
    cv::Size grid_size(point_maps.size(3), point_maps.size(2));
    torch::Tensor center_base = CreateImageGrid(grid_size).to(scores.device());
    center_base.mul_(params_.grid_size_).add_(step);
    torch::Tensor coords =
        center_base + center_shift.mul_(params_.cross_ratio_ * step);

    std::vector<cv::KeyPoint> kpts;
    scores = scores.flatten();
    coords = coords.view({2, -1}).permute({1, 0});
    float scale_height = static_cast<float>(img_size.height) / params_.tensor_height_;
    float scale_width = static_cast<float>(img_size.width) / params_.tensor_width_;
    
    scores = scores.cpu();
    coords = coords.cpu();
    auto scores_data = scores.accessor<float, 1>();
    auto coords_data = coords.accessor<float, 2>();

    cv::Mat mask = detect_mask.getMat();
    for (int64_t i = 0; i < coords.size(0); ++i) {

        float point_score = scores_data[i];
        if (point_score < params_.keypoints_threshold_) {
            continue;
        }
        cv::Point2f tmp_coord(coords_data[i][1], coords_data[i][0]);
        tmp_coord.x = clamp(tmp_coord.x * scale_width, 0, img_size.width - 1);
        tmp_coord.y = clamp(tmp_coord.y * scale_height, 0, img_size.height - 1);
        
        if (mask.empty() || !(mask.at<uchar>(int(tmp_coord.y + 0.5), int(tmp_coord.x + 0.5)))) {
            kpts.emplace_back(tmp_coord, 1, -1.0f, point_score);
        }
    }

    utils::AdaptiveNMS nms(params_.nms_type_);
    keypoints = nms.run(kpts, params_.num_keypoints_, 0.1, img_size.width, img_size.height);
}   


void LiKL::ComputeDescriptors(
            const torch::Tensor& desc_map,
            const std::vector<cv::KeyPoint>& keypoints,
            const cv::Size& img_size,
            cv::Mat& descriptors) const {
    int num_keypoints = keypoints.size();
    if (num_keypoints == 0) {
        descriptors.release();
        return;
    }
    // Create grid
    std::vector<float> grid_vec;
    
    grid_vec.reserve(keypoints.size() * 2);
    for (size_t i = 0; i < keypoints.size(); ++i) {
        float grid_x = keypoints[i].pt.x * 2 / (img_size.width - 1);
        float grid_y = keypoints[i].pt.y * 2 / (img_size.height - 1);
        grid_vec.emplace_back(grid_x - 1);
        grid_vec.emplace_back(grid_y - 1);
    }
    auto ops = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor grid_tensor = torch::from_blob(grid_vec.data(), {1, num_keypoints, 1, 2}, ops);

    _ComputeDescriptorsImpl(desc_map, grid_tensor, descriptors);
}


void LiKL::ComputeDescriptors(
            const torch::Tensor& desc_map,
            const std::vector<cv::Vec2f>& keypoints,
            const cv::Size& img_size,
            cv::Mat& descriptors) const {
    if (keypoints.size() == 0) {
        descriptors.release();
        return;
    }

    int num_keypoints = keypoints.size();
    auto grid_vec = keypoints;
    std::vector<float> factor = {2.0f / (img_size.width - 1), 2.0f / (img_size.height - 1)};

    auto ops = torch::TensorOptions().dtype(torch::kFloat);
    torch::Tensor grid_tensor = torch::from_blob(grid_vec.data(), {1, num_keypoints, 1, 2}, ops);
    torch::Tensor factor_tensor = torch::from_blob(factor.data(), {2}, ops);
    grid_tensor = grid_tensor * factor_tensor - 1;

    _ComputeDescriptorsImpl(desc_map, grid_tensor, descriptors);
}


void LiKL::_ComputeDescriptorsImpl(
            const torch::Tensor& desc_map,
            const torch::Tensor& grid_tensor,
            cv::Mat& descriptors) const {
    namespace F = torch::nn::functional;
    auto sample_ops = F::GridSampleFuncOptions().mode(torch::kBilinear).align_corners(true);
    // [1, D, num_keypoints, 1] ==> [num_keypoints, D]
    torch::Tensor descs = F::grid_sample(desc_map, grid_tensor.to(desc_map.device()), sample_ops);
    int num_keypoints = descs.size(2);
    descs = descs.permute({0, 2, 3, 1}).reshape({num_keypoints, -1});
    descs = torch::nn::functional::normalize(descs, F::NormalizeFuncOptions().p(2).dim(1));
    
    descs = descs.cpu();
    descriptors = cv::Mat(cv::Size(descs.size(1), descs.size(0)), 
                          CV_32FC1, 
                          descs.contiguous().data_ptr<float>()).clone();
}

void LiKL::GetLines(
        const torch::Tensor& line_maps,
        const cv::Size& img_size,
        std::vector<cv::Vec4f>& lines) const {
    torch::Tensor center_map = torch::sigmoid(line_maps.slice(1, 0, 1));
    torch::Tensor displacement_map = line_maps.slice(1, 1, 5);

    torch::Tensor center_nms_map = SimpleNms(center_map);
    auto scores_indices = torch::topk(center_nms_map.reshape(-1), 500);
    torch::Tensor scores = std::get<0>(scores_indices);
    torch::Tensor indices = std::get<1>(scores_indices);

    std::vector<torch::Tensor> valid = torch::where(scores >= params_.line_threshold_);
    scores = scores.index({valid[0]});
    indices = indices.index({valid[0]});

    int w = line_maps.size(3);
    torch::Tensor center_y = indices.div(w, "floor");
    torch::Tensor center_x = indices.fmod(w);
    torch::Tensor center_pos = torch::cat({center_y.unsqueeze(1), center_x.unsqueeze(1)}, 1);

    using namespace torch::indexing;
    torch::Tensor start_points = center_pos + 
        displacement_map.index({0, Slice(None, 2), center_y, center_x})
            .permute({1, 0});
    torch::Tensor end_points = center_pos + 
        displacement_map.index({0, Slice(2, 4), center_y, center_x})
            .permute({1, 0});
    
    float scale_height = static_cast<float>(img_size.height) / params_.tensor_height_;
    float scale_width = static_cast<float>(img_size.width) / params_.tensor_width_;

    // downsample factor: 2
    start_points.select(1, 0).mul_(2 * scale_height).clamp_(0, img_size.height - 1);
    start_points.select(1, 1).mul_(2 * scale_width).clamp_(0, img_size.width - 1);
    end_points.select(1, 0).mul_(2 * scale_height).clamp_(0, img_size.height - 1);
    end_points.select(1, 1).mul_(2 * scale_width).clamp_(0, img_size.width - 1);

    // Filter using line length
    float min_length = params_.min_line_length_;
    if (min_length < 1.0) {
        min_length *= std::min(img_size.height, img_size.width);
    }
    if (min_length > 0) {
        torch::Tensor line_length = torch::norm(start_points - end_points, 2, {-1});
        std::vector<torch::Tensor> length_valid = torch::where(line_length > min_length);
        center_pos = center_pos.index({length_valid[0]});
        start_points = start_points.index({length_valid[0]});
        end_points  = end_points.index({length_valid[0]});        
    }

    start_points = start_points.cpu();
    end_points = end_points.cpu();
    auto start_points_data = start_points.accessor<float, 2>();
    auto end_points_data = end_points.accessor<float, 2>();

    lines.clear();
    lines.reserve(center_pos.size(0));

    for (int64_t i = 0; i < center_pos.size(0); ++i) {
        // xy format
        lines.emplace_back(start_points_data[i][1], start_points_data[i][0],
                           end_points_data[i][1], end_points_data[i][0]);
    }
}


torch::Tensor LiKL::CreateImageGrid(const cv::Size& img_size) const {
    torch::Tensor xs = torch::linspace(0, img_size.width - 1, img_size.width);
    torch::Tensor ys = torch::linspace(0, img_size.height - 1, img_size.height);
    auto grid = torch::meshgrid({ys, xs}, "ij");
    torch::Tensor img_grid = torch::stack({grid[0], grid[1]}, 0);
    return img_grid;
}


} // namespace likl 