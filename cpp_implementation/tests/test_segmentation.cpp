#include "omr/segmentation.h"
#include "test_utils.h"

int main() {
    cv::Mat img = cv::Mat::zeros(100, 100, CV_8UC1);
    cv::rectangle(img, cv::Rect(10, 10, 8, 6), cv::Scalar(255), -1);
    cv::rectangle(img, cv::Rect(50, 50, 8, 6), cv::Scalar(255), -1);
    // too small
    cv::rectangle(img, cv::Rect(80, 80, 2, 2), cv::Scalar(255), -1);

    auto boxes = omr::detectNoteheads(img, 8.0, 6.0);

    // check filtering + width height attributes
    CHECK(boxes.size() == 2);
    for (const auto &b: boxes) {
        CHECK(b.width() == 8);
        CHECK(b.height() == 6);
    }

    return 0;
}
