#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace omr {
    // Detects staff line y-coordinates
    std::vector<int> detectStaffLines(const cv::Mat &imgBin);

    // Groups consecutive staff lines into staves
    std::vector<std::vector<int> > groupStaves(const std::vector<int> &staffLineYs);
}
