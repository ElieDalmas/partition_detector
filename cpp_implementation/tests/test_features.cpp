#include "omr/features.h"
#include "test_utils.h"

int main() {
    cv::Mat plane = cv::Mat::zeros(20, 20, CV_8UC1);
    cv::circle(plane, cv::Point(10, 10), 8, cv::Scalar(255), -1);

    auto feet = omr::extractFeatures(plane);

    // check on features
    CHECK(feet.size() == 11);

    double density = feet[7];
    double aspect = feet[8];
    double centerDensity = feet[9];
    double ringRatio = feet[10];

    CHECK(density > 0.0 && density <= 1.0);
    CHECK(aspect > 0.0);
    CHECK(centerDensity >= 0.0 && centerDensity <= 1.0);
    CHECK(ringRatio >= 0.0 && ringRatio <= 1.0);
    // filled disk -> 1
    CHECK(equal(centerDensity, 1.0, 1e-3));

    return 0;
}
