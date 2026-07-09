#include "omr/otsu.h"
#include "test_utils.h"

int main() {
    // half 10, half 200
    cv::Mat img(8, 8, CV_8UC1, cv::Scalar(200));
    img(cv::Rect(0, 0, 4, 8)).setTo(10);

    auto [thr, mask] = omr::OtsuThreshold::apply(img);

    // check thr in [10, 200]
    CHECK(thr > 10 && thr < 200);
    // check regions
    CHECK(mask.at<uchar>(0, 0) == 255);
    CHECK(mask.at<uchar>(0, 7) == 0);

    return 0;
}
