#pragma once

#include "omr/box.h"

#include <opencv2/opencv.hpp>
#include <vector>

namespace omr {
    // Connected components on the cleaned image, filtered by size/aspect ratio
    std::vector<Box> detectNoteheads(const cv::Mat &imgClean, double avgW, double avgH);
}
