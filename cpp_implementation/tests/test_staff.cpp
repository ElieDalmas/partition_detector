#include "omr/staff.h"
#include "test_utils.h"

#include <cmath>
#include <vector>

int main() {
    // 5 baselines
    cv::Mat img = cv::Mat::zeros(80, 200, CV_8UC1);
    std::vector trueYs = {10, 20, 30, 40, 50};
    for (int y: trueYs) {
        cv::line(img, cv::Point(0, y), cv::Point(199, y), cv::Scalar(255), 1);
    }

    //check  baselines detection
    auto detected = omr::detectStaffLines(img);
    CHECK(detected.size() == 5);
    for (std::size_t i = 0; i < detected.size(); ++i) {
        CHECK(std::abs(detected[i] - trueYs[i]) <= 1);
    }

    // check grouping: staves + barlines size
    auto staves = omr::groupStaves(detected);
    CHECK(staves.size() == 1);
    CHECK(staves[0].size() == 5);

    // check grouping: 2 staves
    std::vector twoStaves = {10, 20, 30, 40, 50, 200, 210, 220, 230, 240};
    auto groups = omr::groupStaves(twoStaves);
    CHECK(groups.size() == 2);

    return 0;
}
