#include "omr/knn.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace omr {
    KNeighborsClassifier::KNeighborsClassifier(std::size_t nNeighbors)
        : _nNeighbors(nNeighbors) {
        if (_nNeighbors < 1) {
            throw std::invalid_argument("n_neighbors >= 1");
        }
    }

    KNeighborsClassifier &KNeighborsClassifier::fit(const std::vector<std::vector<double> > &X,
                                                    const std::vector<std::string> &y) {
        _xTrain = X;
        _yTrain = y;

        std::vector<std::string> labelsSet;
        for (const auto &label: _yTrain) {
            if (std::find(labelsSet.begin(), labelsSet.end(), label) == labelsSet.end()) {
                labelsSet.push_back(label);
            }
        }

        // sklearn sorts classes, so be it
        _classes = labelsSet;
        std::sort(_classes.begin(), _classes.end());
        return *this;
    }

    double KNeighborsClassifier::euclideanDistance(const std::vector<double> &a, const std::vector<double> &b) const {
        double s = 0.0;
        std::size_t len = a.size();
        for (std::size_t f = 0; f < len; ++f) {
            double diff = a[f] - b[f];
            s += diff * diff;
        }
        return std::sqrt(s);
    }

    std::vector<std::string> KNeighborsClassifier::kNearestLabels(const std::vector<double> &query) const {
        std::size_t k = _nNeighbors;
        std::size_t nTrain = _xTrain.size();

        // list[(distance, label)], placeholder: +inf, ""
        std::vector<std::pair<double, std::string> > best(k, {std::numeric_limits<double>::infinity(), ""});

        for (std::size_t i = 0; i < nTrain; ++i) {
            double d = euclideanDistance(query, _xTrain[i]);

            // sort d if lower than last
            if (d < best.back().first) {
                best.back() = {d, _yTrain[i]};

                // partial insertion sort
                long j = static_cast<long>(k) - 1;
                while (j > 0 && best[j].first < best[j - 1].first) {
                    std::swap(best[j], best[j - 1]);
                    --j;
                }
            }
        }

        std::vector<std::string> labels;
        labels.reserve(k);
        for (const auto &[dist, label]: best) {
            labels.push_back(label);
        }
        return labels;
    }

    std::string KNeighborsClassifier::majorityVote(const std::vector<std::string> &labels) const {
        std::vector<std::pair<std::string, int> > counts;

        for (const auto &label: labels) {
            auto it = std::find_if(counts.begin(), counts.end(), [&](const auto &p) { return p.first == label; });

            if (it == counts.end()) {
                counts.emplace_back(label, 1);
            } else {
                it->second += 1;
            }
        }

        // find best class
        std::string bestLabel;
        int bestCount = -1;
        for (const auto &[label, count]: counts) {
            if (count > bestCount) {
                bestCount = count;
                bestLabel = label;
            }
        }
        return bestLabel;
    }

    std::vector<std::string> KNeighborsClassifier::predict(const std::vector<std::vector<double> > &X) const {
        checkIsFitted();

        std::vector<std::string> predictions;
        predictions.reserve(X.size());
        for (const auto &query: X) {
            auto neighborLabels = kNearestLabels(query);
            predictions.push_back(majorityVote(neighborLabels));
        }
        return predictions;
    }

    std::vector<std::vector<double> > KNeighborsClassifier::predictProba(
        const std::vector<std::vector<double> > &X) const {
        checkIsFitted();

        std::size_t nClasses = _classes.size();
        std::vector proba(X.size(), std::vector(nClasses, 0.0));

        for (std::size_t i = 0; i < X.size(); ++i) {
            auto neighborLabels = kNearestLabels(X[i]);
            for (const auto &label: neighborLabels) {
                auto it = std::find(_classes.begin(), _classes.end(), label);
                if (it != _classes.end()) {
                    std::size_t classIdx = static_cast<std::size_t>(std::distance(_classes.begin(), it));
                    proba[i][classIdx] += 1.0;
                }
            }
            for (std::size_t c = 0; c < nClasses; ++c) {
                proba[i][c] /= static_cast<double>(_nNeighbors);
            }
        }

        return proba;
    }

    void KNeighborsClassifier::checkIsFitted() const {
        if (_xTrain.empty()) {
            throw std::runtime_error("Classifier not fitted. Call fit() first.");
        }
    }
}