#pragma once

#include "omr/box.h"

#include <set>
#include <vector>

namespace omr {
    bool centerInBox(const Box &det, const Box &gt);

    struct EvaluationResult {
        int tp = 0;
        int fp = 0;
        int fn = 0;
        double precision = 0.0;
        double recall = 0.0;
        double f1 = 0.0;
        std::set<int> matched;
    };

    // Center-in-box detection evaluation
    EvaluationResult evaluate(const std::vector<Box> &detections, const std::vector<Box> &gtBoxes);

    double computeIou(const Box &b1, const Box &b2);

    // F1 score at a fixed IoU>=0.5 matching threshold
    double apAt05(const std::vector<Box> &dets, const std::vector<Box> &gts);
}
