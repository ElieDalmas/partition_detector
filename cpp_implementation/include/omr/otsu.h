#pragma once

#include <opencv2/opencv.hpp>
#include <array>
#include <utility>

namespace omr {
    // Otsu threshold with inverse binarisation
    //
    // Default:
    // - max val: 255
    // - mode: BINARY_INV (255 if pixel <= thr else 0)
    class OtsuThreshold {
    public:
        // return {threshold, mask}
        std::pair<int, cv::Mat> threshold(const cv::Mat &img) const;

        static std::pair<int, cv::Mat> apply(const cv::Mat &img);

    private:
        static std::array<long, 256> buildHistogram(const cv::Mat &img);

        // Maximize inter-class variance
        static int computeOtsuThreshold(const cv::Mat &img);

        static cv::Mat check(const cv::Mat &img);

        // 255 if pixel <= thresh else 0  (THRESH_BINARY_INV)
        static cv::Mat applyBinaryInv(const cv::Mat &img, int thresh);
    };
}
