#pragma once

#include "omr/box.h"

#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>

namespace omr {
    // Parse "instance:#000042;duration:8;rel_position:-4;..."
    // return rel_position value
    std::optional<int> parseRelPosition(const std::string &comments);

    // Y center of mass (in pixel-space) of the foreground pixels inside bbox
    double getCyFromPixels(const Box &bbox, const cv::Mat &imgBin);

    // Map notehead y center to staff relative position (default: SolG, 5-line staff)
    // include 3 ledger lines above/below.
    int relPositionFromY(double noteCy, const std::vector<int> &staffLinesYs);
}
