#include "omr/evaluation.h"

#include <algorithm>

namespace omr {
    bool centerInBox(const Box &det, const Box &gt) {
        double cx = (det.x1 + det.x2) / 2.0;
        double cy = (det.y1 + det.y2) / 2.0;
        return gt.x1 <= cx && cx <= gt.x2 && gt.y1 <= cy && cy <= gt.y2;
    }

    EvaluationResult evaluate(const std::vector<Box> &detections, const std::vector<Box> &gtBoxes) {
        EvaluationResult res;

        for (const auto &det: detections) {
            bool hit = false;
            for (std::size_t i = 0; i < gtBoxes.size(); ++i) {
                if (res.matched.count(static_cast<int>(i)) == 0 && centerInBox(det, gtBoxes[i])) {
                    res.tp++;
                    res.matched.insert(static_cast<int>(i));
                    hit = true;
                    break;
                }
            }
            if (!hit) {
                res.fp++;
            }
        }

        // compute metrics
        res.fn = static_cast<int>(gtBoxes.size()) - static_cast<int>(res.matched.size());
        res.precision = (res.tp + res.fp) > 0
                               ? static_cast<double>(res.tp) / (res.tp + res.fp)
                               : 0.0;
        res.recall = (res.tp + res.fn) > 0
                            ? static_cast<double>(res.tp) / (res.tp + res.fn)
                            : 0.0;
        res.f1 = (res.precision + res.recall) > 0
                        ? 2 * res.precision * res.recall / (res.precision + res.recall)
                        : 0.0;

        return res;
    }

    double computeIou(const Box &b1, const Box &b2) {
        int xi1 = std::max(b1.x1, b2.x1);
        int yi1 = std::max(b1.y1, b2.y1);
        int xi2 = std::min(b1.x2, b2.x2);
        int yi2 = std::min(b1.y2, b2.y2);

        double inter = static_cast<double>(std::max(0, xi2 - xi1)) * static_cast<double>(std::max(0, yi2 - yi1));
        double union_ = static_cast<double>(b1.width()) * b1.height() +
                        static_cast<double>(b2.width()) * b2.height() - inter;

        return union_ > 0 ? inter / union_ : 0.0;
    }

    double apAt05(const std::vector<Box> &dets, const std::vector<Box> &gts) {
        std::set<int> matched;
        int tp = 0;
        int fp = 0;

        for (const auto &det: dets) {
            int best = -1;
            double bestIou = -1.0;
            for (std::size_t i = 0; i < gts.size(); ++i) {
                double iou = computeIou(det, gts[i]);
                if (iou > bestIou) {
                    bestIou = iou;
                    best = static_cast<int>(i);
                }
            }

            if (best >= 0 && bestIou >= 0.5 && matched.count(best) == 0) {
                tp++;
                matched.insert(best);
            } else {
                fp++;
            }
        }

        int fn = static_cast<int>(gts.size()) - static_cast<int>(matched.size());
        double p = (tp + fp) > 0 ? static_cast<double>(tp) / (tp + fp) : 0.0;
        double r = (tp + fn) > 0 ? static_cast<double>(tp) / (tp + fn) : 0.0;
        return (p + r) > 0 ? 2 * p * r / (p + r) : 0.0;
    }
}
