#include "omr/otsu.h"

#include <stdexcept>
#include <sstream>

namespace omr {

std::pair<int, cv::Mat> OtsuThreshold::threshold(const cv::Mat& img) const {
    cv::Mat checked = check(img);
    int thr = computeOtsuThreshold(checked);
    cv::Mat mask = applyBinaryInv(checked, thr);
    return {thr, mask};
}

std::pair<int, cv::Mat> OtsuThreshold::apply(const cv::Mat& img) {
    return OtsuThreshold().threshold(img);
}

std::array<long, 256> OtsuThreshold::buildHistogram(const cv::Mat& img) {
    std::array<long, 256> hist{};
    hist.fill(0);
    for (int r = 0; r < img.rows; ++r) {
        const uchar* row = img.ptr<uchar>(r);
        for (int c = 0; c < img.cols; ++c) {
            hist[row[c]]++;
        }
    }
    return hist;
}

int OtsuThreshold::computeOtsuThreshold(const cv::Mat& img) {
    int level = 0;
    auto hist = buildHistogram(img);
    long total = static_cast<long>(img.rows) * static_cast<long>(img.cols);

    long sum_b = 0;
    long w_b = 0;
    double maximum = 0.0;

    long sum1 = 0;
    for (int i = 0; i < 256; ++i) {
        sum1 += static_cast<long>(i) * hist[i];
    }

    for (int ii = 0; ii < 256; ++ii) {
        w_b += hist[ii];
        long w_f = total - w_b;
        sum_b += static_cast<long>(ii) * hist[ii];

        if (w_b > 0 && w_f > 0) {
            double m_f = static_cast<double>(sum1 - sum_b) / static_cast<double>(w_f);
            double m_b = static_cast<double>(sum_b) / static_cast<double>(w_b);
            double val = static_cast<double>(w_b) * static_cast<double>(w_f) * (m_b - m_f) * (m_b - m_f);

            if (val >= maximum) {
                level = ii;
                maximum = val;
            }
        }
    }

    return level;
}

cv::Mat OtsuThreshold::check(const cv::Mat& img) {
    if (img.channels() != 1) {
        std::ostringstream oss;
        oss << "Image must be grayscale (2D). Got channels: " << img.channels();
        throw std::invalid_argument(oss.str());
    }
    if (img.type() != CV_8UC1) {
        cv::Mat converted;
        img.convertTo(converted, CV_8UC1);
        return converted;
    }
    return img;
}

cv::Mat OtsuThreshold::applyBinaryInv(const cv::Mat& img, int thresh) {
    cv::Mat binary(img.rows, img.cols, CV_8UC1);
    for (int r = 0; r < img.rows; ++r) {
        const uchar* srcRow = img.ptr<uchar>(r);
        uchar* dstRow = binary.ptr<uchar>(r);
        for (int c = 0; c < img.cols; ++c) {
            dstRow[c] = (srcRow[c] <= thresh) ? 255 : 0;
        }
    }
    return binary;
}

}