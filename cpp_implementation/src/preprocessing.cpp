#include "omr/preprocessing.h"

#include <algorithm>

namespace omr {

cv::Mat removeStaffBarsAndStems(const cv::Mat& imgBin, double avgNoteheadHeight) {
    int wImg = imgBin.cols;
    int hImg = imgBin.rows;

    // staff lines: long horizontal structures
    cv::Mat kernelHoriz = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(std::max(1, wImg / 4), 1));
    cv::Mat staff;
    cv::erode(imgBin, staff, kernelHoriz);
    cv::dilate(staff, staff, kernelHoriz);
    cv::Mat imgNoStaff;
    cv::subtract(imgBin, staff, imgNoStaff);

    // vertical closing: fill holes in noteheads on-line
    cv::Mat kernelClose = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 3));
    cv::morphologyEx(imgNoStaff, imgNoStaff, cv::MORPH_CLOSE, kernelClose);

    // barlines: tall vertical structures
    cv::Mat kernelVert = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, std::max(1, hImg / 20)));
    cv::Mat bars;
    cv::erode(imgNoStaff, bars, kernelVert);
    cv::dilate(bars, bars, kernelVert);
    cv::Mat imgNoStaffNoBars;
    cv::subtract(imgNoStaff, bars, imgNoStaffNoBars);

    // stems: short vertical structures
    int stemKernelHeight = std::max(1, static_cast<int>(avgNoteheadHeight * 2));
    cv::Mat kernelStem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, stemKernelHeight));
    cv::Mat stems;
    cv::erode(imgNoStaffNoBars, stems, kernelStem);
    cv::dilate(stems, stems, kernelStem);
    cv::Mat imgClean;
    cv::subtract(imgNoStaffNoBars, stems, imgClean);

    return imgClean;
}

}