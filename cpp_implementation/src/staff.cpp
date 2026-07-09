#include "omr/staff.h"

#include <algorithm>
#include <cmath>

namespace omr {
    double median(std::vector<int> values) {
        std::sort(values.begin(), values.end());
        std::size_t n = values.size();
        if (n == 0) return 0.0;
        if (n % 2 == 1) {
            return values[n / 2];
        }
        return (static_cast<double>(values[n / 2 - 1]) + static_cast<double>(values[n / 2])) / 2.0;
    }

    std::vector<int> detectStaffLines(const cv::Mat &imgBin) {
        int wImg = imgBin.cols;

        // short kernel width to avoid interruptions
        cv::Mat kernelStaff = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(std::max(1, wImg / 8), 1));
        cv::Mat staffMask;
        cv::erode(imgBin, staffMask, kernelStaff);
        cv::dilate(staffMask, staffMask, kernelStaff);

        // nbr px on y line
        std::vector hProj(static_cast<std::size_t>(staffMask.rows), 0.0);
        for (int r = 0; r < staffMask.rows; ++r) {
            const uchar *row = staffMask.ptr<uchar>(r);
            long sum = 0;
            for (int c = 0; c < staffMask.cols; ++c) {
                sum += row[c];
            }
            hProj[static_cast<std::size_t>(r)] = static_cast<double>(sum);
        }

        // staff line > 30% width
        double thresholdPx = 0.30 * wImg * 255.0;
        std::vector<int> staffYs;
        for (int r = 0; r < staffMask.rows; ++r) {
            if (hProj[static_cast<std::size_t>(r)] > thresholdPx) {
                staffYs.push_back(r);
            }
        }

        // merge staff line runs on the same line
        std::vector<int> staffLineYs;
        if (!staffYs.empty()) {
            int runStart = staffYs[0];
            int prev = staffYs[0];
            for (std::size_t i = 1; i < staffYs.size(); ++i) {
                int y = staffYs[i];
                // detect gap between two staff lines
                if (y - prev > 3) {
                    staffLineYs.push_back((runStart + prev) / 2);
                    runStart = y;
                }
                prev = y;
            }
            staffLineYs.push_back((runStart + prev) / 2);
        }

        return staffLineYs;
    }

    std::vector<std::vector<int> > groupStaves(const std::vector<int> &staffLineYs) {
        std::vector<std::vector<int> > staves;
        if (staffLineYs.empty()) {
            return staves;
        }

        std::vector<int> diffs;
        for (std::size_t i = 1; i < staffLineYs.size(); ++i) {
            diffs.push_back(staffLineYs[i] - staffLineYs[i - 1]);
        }
        int staffSpacing = diffs.empty() ? 0 : static_cast<int>(median(diffs));

        std::vector current = {staffLineYs[0]};
        for (std::size_t i = 1; i < staffLineYs.size(); ++i) {
            // gap between two consecutive lines exceeds 2x the median line spacing => new staff
            if (staffLineYs[i] - staffLineYs[i - 1] > 2 * staffSpacing) {
                staves.push_back(current);
                current.clear();
            }
            current.push_back(staffLineYs[i]);
        }
        staves.push_back(current);

        return staves;
    }
}
