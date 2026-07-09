#include "omr/pitch.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace omr {
    std::optional<int> parseRelPosition(const std::string &comments) {
        std::stringstream ss(comments);
        std::string part;
        while (std::getline(ss, part, ';')) {
            const std::string prefix = "rel_position:";
            if (part.rfind(prefix, 0) == 0) {
                std::string valueStr = part.substr(prefix.size());
                try {
                    return std::stoi(valueStr);
                } catch (const std::exception &) {
                    return std::nullopt;
                }
            }
        }
        return std::nullopt;
    }

    double getCyFromPixels(const Box &bbox, const cv::Mat &imgBin) {
        cv::Rect roi(bbox.x1, bbox.y1, bbox.width(), bbox.height());
        cv::Rect imgRect(0, 0, imgBin.cols, imgBin.rows);
        roi &= imgRect;

        if (roi.width > 0 && roi.height > 0) {
            cv::Mat crop = imgBin(roi);
            double sumY = 0.0;
            long count = 0;
            for (int r = 0; r < crop.rows; ++r) {
                const uchar *row = crop.ptr<uchar>(r);
                for (int c = 0; c < crop.cols; ++c) {
                    if (row[c] > 0) {
                        sumY += r;
                        ++count;
                    }
                }
            }
            if (count > 0) {
                return bbox.y1 + sumY / static_cast<double>(count);
            }
        }
        return (bbox.y1 + bbox.y2) / 2.0;
    }

    // map rel_position : y
    struct PosMap {
        std::vector<std::pair<int, double> > entries;

        void set(int rel, double y) {
            for (auto &e: entries) {
                if (e.first == rel) {
                    e.second = y;
                    return;
                }
            }
            entries.emplace_back(rel, y);
        }
    };

    int relPositionFromY(double noteCy, const std::vector<int> &staffLinesYs) {
        std::vector<int> sortedYs = staffLinesYs;
        std::sort(sortedYs.begin(), sortedYs.end());

        const int lineRels[5] = {4, 2, 0, -2, -4};
        std::size_t n = sortedYs.size();

        double topSp = (n >= 2) ? static_cast<double>(sortedYs[1] - sortedYs[0]) : 0.0;
        double botSp = (n >= 2) ? static_cast<double>(sortedYs[n - 1] - sortedYs[n - 2]) : 0.0;

        PosMap allPos;

        // Lines
        std::size_t nLines = std::min<std::size_t>(5, n);
        for (std::size_t i = 0; i < nLines; ++i) {
            allPos.set(lineRels[i], sortedYs[i]);
        }

        // Spaces
        // middle between two adjacent lines
        if (n >= 1) {
            for (std::size_t i = 0; i + 1 < n && i < 4; ++i) {
                allPos.set(lineRels[i] - 1, (sortedYs[i] + sortedYs[i + 1]) / 2.0);
            }
        }

        // Ledger lines (3 levels above or below)
        if (n >= 2) {
            for (int d = 1; d <= 3; ++d) {
                allPos.set(4 + d * 2, sortedYs[0] - d * topSp);
                allPos.set(-4 - d * 2, sortedYs[n - 1] + d * botSp);
                allPos.set(4 + d * 2 - 1, sortedYs[0] - (d - 0.5) * topSp);
                allPos.set(-4 - d * 2 + 1, sortedYs[n - 1] + (d - 0.5) * botSp);
            }
        }

        int bestRel = 0;
        double bestDist = std::numeric_limits<double>::infinity();
        for (const auto &[rel, y]: allPos.entries) {
            double dist = std::abs(y - noteCy);
            if (dist < bestDist) {
                bestDist = dist;
                bestRel = rel;
            }
        }
        return bestRel;
    }
}
