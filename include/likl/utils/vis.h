#include <opencv2/core.hpp>

namespace likl {

namespace utils {

inline cv::Scalar GenerateColor(int id) {
    int blue = (id * 123 + 100) % 255;
    int green = (id * 23 + 45) % 255;
    int red = (id * 78 + 156) % 255;
    return cv::Scalar(blue, green, red);
}

void DrawKeypoints(const cv::Mat& img,
                   const std::vector<cv::KeyPoint>& keypoints,
                   int radius, 
                   const cv::Scalar &color, 
                   int thickness = 1, 
                   int lineType = 8, 
                   int shift = 0);

void DrawLines(const cv::Mat& img, 
               const std::vector<cv::Vec4f>& lines,
               const cv::Scalar &color, 
               int thickness = 2, 
               int lineType = 16, 
               int shift = 0);

void DrawLineMatch(const cv::Mat& img1,
                   const std::vector<cv::Vec4f>& lines1,
                   const cv::Mat& img2,
                   const std::vector<cv::Vec4f>& lines2,
                   std::vector<std::pair<int, float>> line_matches,
                   cv::OutputArray out_img,
                   int thickness = 2, 
                   int lineType = 16);

}  // namespace utils

}  // namespace likl
