#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "filters/filters.h"
#include "tickmetr/TickMeter.h"

int main() {
TickMeter timer;
size_t windowSize = 3;
cv::Mat input = cv::imread("../files/moleworm.jpg");
timer.start();
cv::Mat customBoxFilter = Filtering::BoxFilter(input, windowSize);
timer.stop();
double microsec_customBoxFilter = timer.getTimeMicro();
timer.reset();
cv::Mat cvBoxFilter;
timer.start();
cv::blur(input, cvBoxFilter, cv::Size(windowSize, windowSize));
timer.stop();
double microsec_cvBoxFilter = timer.getTimeMicro();
timer.reset();
timer.start();
auto diffBoxFilter = Filtering::CalcDiff(customBoxFilter, cvBoxFilter);
timer.stop();
double microsec_diffBoxFilter = timer.getTimeMicro();
timer.reset();
std::cout << "dif = " << diffBoxFilter.second << std::endl;
std::cout << "microsec_customBoxFilter = " << microsec_customBoxFilter <<
std::endl;
std::cout << "microsec_cvBoxFilter = " << microsec_cvBoxFilter << std::endl;
std::cout << "microsec_diffBoxFilter = " << microsec_diffBoxFilter <<
std::endl;
cv::imwrite("../files/customBoxFilter.jpg", customBoxFilter);
cv::imwrite("../files/cvBoxFilter.jpg", cvBoxFilter);
cv::imwrite("../files/diffBoxFilter.jpg", diffBoxFilter.first);
cv::imshow("input", input);
cv::imshow("customBoxFilter", customBoxFilter);
cv::imshow("cvBoxFilter", cvBoxFilter);
cv::imshow("diffBoxFilter", diffBoxFilter.first);
cv::waitKey(-1);
return 0;
}