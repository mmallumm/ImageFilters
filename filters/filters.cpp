#include "filters.h"

#define PI 3.14159265

namespace {

cv::Vec3i CVAbs(cv::Vec3i& value) {
  cv::Vec3i res;
  for (int i = 0; i < 3; ++i) {
    res[i] = abs(value[i]);
  }
  return res;
}

template <class pixel_t, class allignment_t = pixel_t>
pixel_t Average(cv::Mat roi) {
  pixel_t result;
  allignment_t resultAllignment;
  cv::MatIterator_<pixel_t> it, end;

  for (it = roi.begin<pixel_t>(), end = roi.end<pixel_t>(); it != end; ++it) {
    resultAllignment += *it;
  }

  result = resultAllignment / (roi.rows * roi.cols);
  return result;
}

template <class pixel_t, class allignment_t = pixel_t>
pixel_t Gaussian(cv::Mat roi, double sigma) {
  pixel_t result;
  allignment_t resultAllignment;
  double sigmaSum = 0;
  for (int i = 0; i < roi.rows; ++i) {
    for (int j = 0; j < roi.cols; ++j) {
      double w = exp(-(i * i + j * j) / (2 * std::pow(sigma, 2)));
      resultAllignment += w * roi.at<pixel_t>(i, j);
      sigmaSum += w;
    }
  }

  result = resultAllignment / sigmaSum;
  return result;
}

cv::Vec3i Laplasian(cv::Mat roi) {
  cv::Vec3i resultAllignment;
  int x = roi.cols / 2;
  int y = roi.rows / 2;
  cv::Vec3i val_1 = roi.at<cv::Vec3b>(x + 1, y);
  cv::Vec3i val_2 = roi.at<cv::Vec3b>(x - 1, y);
  cv::Vec3i val_3 = roi.at<cv::Vec3b>(x, y + 1);
  cv::Vec3i val_4 = roi.at<cv::Vec3b>(x, y - 1);
  cv::Vec3i val_0 = roi.at<cv::Vec3b>(x, y);
  cv::Vec3i val_res = val_1 + val_2 + val_3 + val_4 - 4 * val_0;
  resultAllignment = val_res;
  return resultAllignment;
}

}  // namespace

cv::Mat Filtering::BoxFilter(cv::Mat image, size_t windowSize) {
  if (windowSize % 2 == 0) {
    return image;
  }

  cv::Mat result = image.clone();
  int shift = windowSize / 2;

  for (int i = shift; i < result.rows - shift; ++i) {
    for (int j = shift; j < result.cols - shift; ++j) {
      cv::Rect window(j - shift, i - shift, windowSize, windowSize);
      switch (result.channels()) {
        case 1:
          result.at<uchar>(i, j) = Average<uchar, uint>(image(window));
          break;
        case 3:
          result.at<cv::Vec3b>(i, j) =
              Average<cv::Vec3b, cv::Vec3i>(image(window));
          break;
        default:
          break;
      }
    }
  }

  return result;
}

cv::Mat Filtering::GaussianFilter(cv::Mat image, double sigma,
                                  size_t windowSize) {
  if (windowSize % 2 == 0) {
    return image;
  }

  cv::Mat result = image.clone();
  int shift = windowSize / 2;

  for (int i = shift; i < result.rows - shift; ++i) {
    for (int j = shift; j < result.cols - shift; ++j) {
      cv::Rect window(j - shift, i - shift, windowSize, windowSize);
      switch (result.channels()) {
        case 1:
          result.at<uchar>(i, j) = Gaussian<uchar, uint>(image(window), sigma);
          break;
        case 3:
          result.at<cv::Vec3b>(i, j) =
              Gaussian<cv::Vec3b, cv::Vec3i>(image(window), sigma);
          break;
        default:
          break;
      }
    }
  }

  return result;
}

cv::Mat Filtering::LaplasianFilter(cv::Mat image, double alpha) {
  cv::Mat result = image.clone();
  int windowSize = 3;
  int shift = windowSize / 2;
  for (int i = shift; i < result.rows - shift; ++i) {
    for (int j = shift; j < result.cols - shift; ++j) {
      cv::Rect window(j - shift, i - shift, windowSize, windowSize);
      switch (result.channels()) {
        case 3: {
          auto laplasian = Laplasian(image(window));
          cv::Vec3b laplasianVec = alpha * laplasian;
          result.at<cv::Vec3b>(i, j) =
              image.at<cv::Vec3b>(i, j) + laplasianVec;
        }
      }
    }
  }

  return result;
}

cv::Mat Filtering::UnsharpMasking(cv::Mat image, cv::Mat smoothed_image, double alpha) {
  cv::Mat result = (image + alpha * (image - smoothed_image));
  return result;
}

std::pair<cv::Mat, double> Filtering::CalcDiff(cv::Mat image_1,
                                               cv::Mat image_2) {
  cv::Mat image_3 = image_1.clone();

  int equalCounter = 0;

  switch (image_1.channels()) {
    case 1: {
      cv::MatIterator_<uchar> it_1, end_1, it_2, it_3;
      for (it_1 = image_1.begin<uchar>(), it_2 = image_2.begin<uchar>(),
          it_3 = image_3.begin<uchar>(), end_1 = image_1.end<uchar>();
           it_1 != end_1; ++it_1, ++it_2) {
        *it_3 = std::abs(static_cast<int>(*it_2 - *it_1));
        equalCounter += (*it_3 == 0) ? 1 : 0;
      }
      break;
    }
    case 3: {
      cv::MatIterator_<cv::Vec3b> it_1, end_1, it_2, it_3;
      for (it_1 = image_1.begin<cv::Vec3b>(), it_2 = image_2.begin<cv::Vec3b>(),
          it_3 = image_3.begin<cv::Vec3b>(), end_1 = image_1.end<cv::Vec3b>();
           it_1 != end_1; ++it_1, ++it_2, ++it_3) {
        for (int c = 0; c < image_3.channels(); ++c) {
          (*it_3)[c] = std::abs(static_cast<int>((*it_2)[c] - (*it_1)[c]));
          equalCounter += (*it_3 == cv::Vec3b(0, 0, 0)) ? 1 : 0;
        }
      }
      break;
    }

    default:
      break;
  }

  double equalPercent = static_cast<double>(equalCounter) /
                        static_cast<double>(image_3.rows * image_3.cols);
  return std::pair<cv::Mat, double>(std::move(image_3), equalPercent);
}

cv::Mat Filtering::LogIntensityCorrection(cv::Mat inputImage, double alpha) {
  cv::Mat image = inputImage.clone();
  switch (image.channels()) {
    case 1: {
      cv::MatIterator_<uchar> it = image.begin<uchar>();
      cv::MatIterator_<uchar> end = image.end<uchar>();
      for (; it != end; ++it) {
        *it = alpha * log(1 + *it);
      }
      break;
    }

    case 3: {
      cv::MatIterator_<cv::Vec3b> it = image.begin<cv::Vec3b>();
      cv::MatIterator_<cv::Vec3b> end = image.end<cv::Vec3b>();
      for (; it != end; ++it) {
        for (int c = 0; c < image.channels(); ++c) {
          (*it)[c] = alpha * log(1 + (*it)[c]);
        }
      }
      break;
    }

    default:
      break;
  }
  return image;
}