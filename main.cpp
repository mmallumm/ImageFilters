#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "filters/filters.h"
#include "tickmetr/TickMeter.h"

int main() {
  double laplase_alpha = -3;
  double alpha = 10;

  cv::Mat input = cv::imread("../files/moleworm.jpg");
  cv::Mat unsharpLaplasian  = Filtering::UnsharpMasking(input, Filtering::LaplasianFilter(input, laplase_alpha), alpha);
  cv::imshow("unsharpLaplasian", unsharpLaplasian);
  cv::waitKey(-1);

  return 0;
}