#include "filters/filters.h"

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

int main() {
    size_t windowSize = 9;
    cv::Mat input = cv::imread("../files/moleworm.jpg");

    cv::Mat customBoxFilter = Filtering::BoxFilter(input, windowSize);
    cv::Mat cvBoxFilter;
    cv::blur(input, cvBoxFilter, cv::Size(windowSize,windowSize));
    auto diffBoxFilter = Filtering::CalcDiff(customBoxFilter, cvBoxFilter);
    cv::imwrite("../files/customBoxFilter.jpg", customBoxFilter);
    cv::imwrite("../files/cvBoxFilter.jpg", cvBoxFilter);
    cv::imwrite("../files/diffBoxFilter.jpg", diffBoxFilter.first);


    cv::Mat customGaussianFilter = Filtering::GaussianFilter(input, 4, windowSize); 
    cv::Mat cvGaussinFilter;
    cv::GaussianBlur(input, cvGaussinFilter, cv::Size(windowSize,windowSize), 4);
    auto diffGaussianBoxFilter = Filtering::CalcDiff(customGaussianFilter, customBoxFilter);
    diffGaussianBoxFilter.first = Filtering::LogIntensityCorrection(diffGaussianBoxFilter.first, 20);
    cv::imwrite("../files/customGaussianFilter.jpg", customGaussianFilter);
    cv::imwrite("../files/cvGaussinFilter.jpg", cvGaussinFilter);
    cv::imwrite("../files/diffGaussianBoxFilter.jpg", diffGaussianBoxFilter.first);

    cv::Mat customLaplasinFilter = Filtering::LaplasianFilter(input, 0.6);
    cv::imwrite("../files/customLaplasinFilter.jpg", customLaplasinFilter);

    cv::Mat customBoxMask = Filtering::UnsharpMasking(input, customBoxFilter, 0.5);
    cv::imwrite("../files/customBoxMask.jpg", customBoxMask);

    return 0;
}