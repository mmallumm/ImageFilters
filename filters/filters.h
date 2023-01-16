#include <opencv2/core.hpp>

namespace Filtering {
cv::Mat BoxFilter(cv::Mat image, size_t windowSize = 3);
cv::Mat GaussianFilter(cv::Mat image, double sigma, size_t windowSize = 3);
cv::Mat LaplasianFilter(cv::Mat image, double alpha);
cv::Mat UnsharpMasking(cv::Mat image, cv::Mat smoothed_image, float alpha);
cv::Mat LogIntensityCorrection(cv::Mat image, double alpha);

std::pair<cv::Mat, double> CalcDiff(cv::Mat image_1, cv::Mat image_2);
}  // namespace Filtering