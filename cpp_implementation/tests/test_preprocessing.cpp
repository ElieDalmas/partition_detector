#include "omr/preprocessing.h"
#include "test_utils.h"

int main() {
    cv::Mat img = cv::Mat::zeros(120, 200, CV_8UC1);
    // 1 staff line
    cv::line(img, cv::Point(0, 10), cv::Point(199, 10), cv::Scalar(255), 1);
    //1 notehead
    cv::rectangle(img, cv::Rect(90, 60, 10, 4), cv::Scalar(255), -1);

    int before = cv::countNonZero(img);
    cv::Mat cleaned = omr::removeStaffBarsAndStems(img, 10.0);
    int after = cv::countNonZero(cleaned);

    // check remove staff line, but keep notehead
    CHECK(cleaned.size() == img.size());
    CHECK(after < before);
    CHECK(after > 0);

    return 0;
}
