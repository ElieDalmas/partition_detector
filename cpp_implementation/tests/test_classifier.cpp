#include "omr/classifier.h"
#include "test_utils.h"

int main() {
    // check level 1 labels
    CHECK(omr::level1Label("noteheadBlackOnLine") == "notehead");
    CHECK(omr::level1Label("restQuarter") == "rest");
    CHECK(omr::level1Label("clefG") == "clef");
    CHECK(omr::level1Label("somethingElse") == "other");

    // check level 2 labels
    CHECK(omr::level2Label("noteheadBlackInSpace").value() == "black");
    CHECK(omr::level2Label("noteheadHalfOnLine").value() == "half");
    CHECK(omr::level2Label("noteheadWholeInSpace").value() == "whole");
    CHECK(!omr::level2Label("restQuarter").has_value());

    // check annotation state after building training data
    // must skip annotations with w or h outside [4, 200]
    std::unordered_map<std::string, std::string> idToName = {{"1", "noteheadBlackOnLine"}};
    omr::Annotation tooSmall;
    tooSmall.catId = {"1"};
    tooSmall.aBbox = {0, 0, 2, 2};

    omr::Annotation ok;
    ok.catId = {"1"};
    ok.aBbox = {0, 0, 10, 10};

    cv::Mat imgBin = cv::Mat::zeros(50, 50, CV_8UC1);
    cv::rectangle(imgBin, cv::Rect(0, 0, 10, 10), cv::Scalar(255), -1);

    auto data = omr::buildTrainingData({tooSmall, ok}, idToName, imgBin);
    CHECK(data.X.size() == 1);
    CHECK(data.yL1[0] == "notehead");
    CHECK(data.yL2[0].has_value() && data.yL2[0].value() == "black");

    return 0;
}
