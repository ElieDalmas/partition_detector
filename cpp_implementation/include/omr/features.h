#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace omr {
    // Features vector for a symbol: 7 Hu moments + density + aspect ratio + center density + ring ratio
    // 11 features
    std::vector<double> extractFeatures(const cv::Mat &mat);
}
