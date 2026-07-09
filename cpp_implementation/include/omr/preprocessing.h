#pragma once

#include <opencv2/opencv.hpp>

namespace omr {
    // Removes staff lines, barlines and stems from a binarized image
    cv::Mat removeStaffBarsAndStems(const cv::Mat &imgBin, double avgNoteheadHeight);
}
