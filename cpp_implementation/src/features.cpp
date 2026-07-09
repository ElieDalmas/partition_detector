#include "omr/features.h"

#include <algorithm>
#include <cmath>

namespace omr {
    std::vector<double> extractFeatures(const cv::Mat &mat) {
        cv::Mat matResized;
        cv::resize(mat, matResized, cv::Size(32, 32));

        cv::Moments moments = cv::moments(matResized, false);
        double huRaw[7];
        cv::HuMoments(moments, huRaw);

        std::vector<double> hu(7);
        for (int i = 0; i < 7; ++i) {
            double sign = (huRaw[i] > 0.0) - (huRaw[i] < 0.0);
            hu[i] = -sign * std::log10(std::abs(huRaw[i]) + 1e-10);
        }

        int h = mat.rows;
        int w = mat.cols;
        // avoid 0 div
        double density = static_cast<double>(cv::countNonZero(mat)) /
                         (static_cast<double>(h) * static_cast<double>(w) + 1e-6);
        double aspect = static_cast<double>(w) / (static_cast<double>(h) + 1e-6);

        // center crop
        int ch = h / 4;
        int cw = w / 4;
        int cy0 = std::clamp(h / 2 - ch, 0, h);
        int cy1 = std::clamp(h / 2 + ch, 0, h);
        int cx0 = std::clamp(w / 2 - cw, 0, w);
        int cx1 = std::clamp(w / 2 + cw, 0, w);

        double cDensity = 0.0;
        if (cy1 > cy0 && cx1 > cx0) {
            cv::Mat c = mat(cv::Range(cy0, cy1), cv::Range(cx0, cx1));
            double cSize = static_cast<double>(c.rows) * static_cast<double>(c.cols);
            cDensity = static_cast<double>(cv::countNonZero(c)) / (cSize + 1e-6);
        }

        cv::Mat inner = cv::Mat::zeros(32, 32, CV_8UC1);
        cv::ellipse(inner, cv::Point(16, 16), cv::Size(8, 6), 0, 0, 360, cv::Scalar(255), -1);
        cv::Mat border = matResized.clone();
        border.setTo(0, inner > 0);
        double ringRatio = static_cast<double>(cv::countNonZero(border)) /
                           (static_cast<double>(cv::countNonZero(matResized)) + 1e-6);

        std::vector<double> features;
        features.reserve(11);
        features.insert(features.end(), hu.begin(), hu.end());
        features.push_back(density);
        features.push_back(aspect);
        features.push_back(cDensity);
        features.push_back(ringRatio);
        return features;
    }
}
