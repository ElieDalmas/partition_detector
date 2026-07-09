#include "omr/evaluation.h"
#include "test_utils.h"

int main() {
    omr::Box gt{0, 0, 10, 10};
    // inside gt
    omr::Box detInside{2, 2, 8, 8};
    omr::Box detOutside{20, 20, 30, 30};

    // check centerInBox function
    CHECK(omr::centerInBox(detInside, gt));
    CHECK(!omr::centerInBox(detOutside, gt));

    // check metrics
    std::vector gts = {gt};
    std::vector dets = {detInside, detOutside};
    auto result = omr::evaluate(dets, gts);
    CHECK(result.tp == 1);
    CHECK(result.fp == 1);
    CHECK(result.fn == 0);
    CHECK(equal(result.precision, 0.5));
    CHECK(equal(result.recall, 1.0));

    // check IoU results
    CHECK(equal(omr::computeIou(gt, gt), 1.0));
    CHECK(equal(omr::computeIou(gt, detOutside), 0.0));

    // check ap metric
    double ap = omr::apAt05(dets, gts);
    CHECK(ap >= 0.0 && ap <= 1.0);

    return 0;
}
