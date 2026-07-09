#include "omr/segmentation.h"

namespace omr {
    std::vector<Box> detectNoteheads(const cv::Mat &imgClean, double avgW, double avgH) {
        cv::Mat labels, stats, centroids;
        int numLabels = cv::connectedComponentsWithStats(imgClean, labels, stats, centroids);

        std::vector<Box> detected;
        for (int i = 1; i < numLabels; ++i) {
            int x = stats.at<int>(i, cv::CC_STAT_LEFT);
            int y = stats.at<int>(i, cv::CC_STAT_TOP);
            int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
            int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
            int area = stats.at<int>(i, cv::CC_STAT_AREA);

            double aspect = h > 0 ? static_cast<double>(w) / h : 0.0;
            if (aspect > 0.5 && aspect < 2.0 &&
                w > avgW * 0.4 && w < avgW * 2.0 &&
                h < avgH * 2.0 && area > 15) {
                detected.push_back(Box{x, y, x + w, y + h});
            }
        }

        return detected;
    }
}
