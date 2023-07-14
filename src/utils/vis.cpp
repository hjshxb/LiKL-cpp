#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "likl/utils/vis.h"


namespace likl {

namespace utils {

void DrawKeypoints(const cv::Mat& img, 
                   const std::vector<cv::KeyPoint>& keypoints,
                   int radius, 
                   const cv::Scalar &color, 
                   int thickness, 
                   int lineType, 
                   int shift) {
    for (size_t i = 0; i < keypoints.size(); ++i) {
        cv::circle(img, keypoints[i].pt, radius, color, thickness, lineType, shift);
    }
}

void DrawLines(const cv::Mat& img, 
               const std::vector<cv::Vec4f>& lines,
               const cv::Scalar &color, 
               int thickness, 
               int lineType, 
               int shift) {
    for (size_t i = 0; i < lines.size(); ++i) {
        cv::line(img,
                 cv::Point(lines[i][0], lines[i][1]),
                 cv::Point(lines[i][2], lines[i][3]),
                 color,
                 thickness,
                 lineType,
                 shift);
    }
}

void DrawLineMatch(const cv::Mat& img1,
                   const std::vector<cv::Vec4f>& lines1,
                   const cv::Mat& img2,
                   const std::vector<cv::Vec4f>& lines2,
                   std::vector<std::pair<int, float>> line_matches,
                   cv::OutputArray out_img,
                   int thickness, 
                   int lineType
                   ) {
    cv::Size out_size(img1.cols + img2.cols, std::max(img1.rows, img2.rows));
    out_img.create(out_size, CV_8UC3);
    cv::Mat out_img_mat = out_img.getMat();
    cv::Mat roi1 = out_img_mat(cv::Rect(0, 0, img1.cols, img1.rows));
    cv::Mat roi2 = out_img_mat(cv::Rect(img1.cols, 0, img2.cols, img2.rows));
    img1.copyTo(roi1);
    img2.copyTo(roi2);

    // Draw match lines
    for (size_t i = 0; i < line_matches.size(); ++i) {
        if (line_matches[i].first == -1) continue;
        int match_idx = line_matches[i].first;
        CV_Assert(i >=0 && i < lines1.size());
        CV_Assert(match_idx >=0 && match_idx < static_cast<int>(lines2.size()));

        cv::Scalar color = likl::utils::GenerateColor(i);
        cv::line(roi1,
                 cv::Point(lines1[i][0], lines1[i][1]),
                 cv::Point(lines1[i][2], lines1[i][3]),
                 color,
                 thickness,
                 lineType);
        cv::line(roi2,
                 cv::Point(lines2[match_idx][0], lines2[match_idx][1]),
                 cv::Point(lines2[match_idx][2], lines2[match_idx][3]),
                 color,
                 thickness,
                 lineType);
    }
}

} //namespace utils
} //namespace likl