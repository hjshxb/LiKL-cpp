#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <torch/cuda.h>

#include "likl/config.h"
#include "likl/feature/line_matcher.h"
#include "likl/feature/likl_extractor.h"
#include "likl/utils/timer.h"
#include "likl/utils/vis.h"

int main(int argc, char** argv) {
    cv::Mat img1 = cv::imread(argv[1]);
    cv::Mat img2 = cv::imread(argv[2]);

    // Load model
    likl::Config<likl::FeatureParams> feature_config_("configs/likl.yaml",
                                                      "Feature");
    auto device = (torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU;
    likl::LiKL likl_extractor(feature_config_.params_, device);
    likl_extractor.WarmUp();

    cv::Mat input_img;
    cv::cvtColor(img1, input_img, cv::COLOR_BGR2RGB);
    cv::Mat input_img2;
    cv::cvtColor(img2, input_img2, cv::COLOR_BGR2RGB);

    std::vector<cv::Mat> vec_img = {input_img, input_img2};
    std::vector<std::vector<cv::KeyPoint>> vec_keypoints;
    std::vector<cv::Mat> vec_desc;
    std::vector<std::vector<cv::Vec4f>> vec_lines;
    std::vector<cv::Mat> vec_line_desc;
    std::vector<cv::Mat> vec_line_mask;

    cv::Mat detect_mask = cv::Mat::zeros(input_img.size(), CV_8UC1);
    detect_mask.rowRange(0, 100) = 1;
    std::vector<cv::Mat> vec_detect_mask = {cv::Mat(), cv::Mat()};

    auto t1 = likl::utils::Timer::tic();
    likl_extractor.Detect(vec_img,
                vec_keypoints,
                vec_desc,
                vec_lines,
                vec_line_desc,
                vec_line_mask,
                vec_detect_mask);
    auto time = likl::utils::Timer::toc(t1);
    std::cout << "lkle detect time = " << time.count() << " ms " << std::endl;

    cv::Mat feature_img(cv::Size(input_img.cols * 2, input_img.rows), CV_8UC3);
    cv::Mat feature_img_roi1 = feature_img.colRange(0, input_img.cols);
    input_img.copyTo(feature_img_roi1);
    likl::utils::DrawLines(
        feature_img_roi1, vec_lines[0], cv::Scalar(0, 255, 0));
    likl::utils::DrawKeypoints(
        feature_img_roi1, vec_keypoints[0], 1, cv::Scalar(0, 0, 255));

    cv::Mat feature_img_roi2 =
        feature_img.colRange(input_img.cols, 2 * input_img.cols);
    input_img2.copyTo(feature_img_roi2);
    likl::utils::DrawLines(
        feature_img_roi2, vec_lines[1], cv::Scalar(0, 255, 0));
    likl::utils::DrawKeypoints(
        feature_img_roi2, vec_keypoints[1], 1, cv::Scalar(0, 0, 255));

    // Match points
    cv::Ptr<cv::BFMatcher> l2_matcher =
        cv::BFMatcher::create(cv::NORM_L2, true);
    std::vector<cv::DMatch> lkle_matches;
    l2_matcher->match(vec_desc[0], vec_desc[1], lkle_matches);
    cv::Mat point_match_img;
    cv::drawMatches(img1,
                    vec_keypoints[0],
                    img2,
                    vec_keypoints[1],
                    lkle_matches,
                    point_match_img);

    // Match lines
    likl::WunschLineMatcher line_matcher(true);
    std::vector<cv::Vec4f>& lines1 = vec_lines[0];
    std::vector<cv::Vec4f>& lines2 = vec_lines[1];

    t1 = likl::utils::Timer::tic();
    std::vector<std::pair<int, float>> line_matches;
    line_matcher.Match(vec_line_desc[0],
                       vec_line_desc[1],
                       vec_line_mask[0],
                       vec_line_mask[1],
                       line_matches);
    time = likl::utils::Timer::toc(t1);
    std::cout << "line match time = " << time.count() << " ms " << std::endl;

    // Draw line matches
    cv::Mat line_match_img;
    likl::utils::DrawLineMatch(
        img1, lines1, img2, lines2, line_matches, line_match_img);

    cv::imshow("features", feature_img);
    cv::imshow("point match", point_match_img);
    cv::imshow("line match", line_match_img);
    cv::waitKey();
    return 0;
}