#pragma once

#include <string>
#include <vector>

namespace omr {
    // default:
    // - weights: majority vote
    // - metric: euclidean
    class KNeighborsClassifier {
    public:
        explicit KNeighborsClassifier(std::size_t nNeighbors = 5);

        KNeighborsClassifier &fit(const std::vector<std::vector<double> > &X, const std::vector<std::string> &y);

        // return corresponding class
        std::vector<std::string> predict(const std::vector<std::vector<double> > &X) const;

        // return [n_samples][n_classes] probabilities
        std::vector<std::vector<double> > predictProba(const std::vector<std::vector<double> > &X) const;

        // get
        const std::vector<std::string> &classes() const { return _classes; }

        // get
        std::size_t nNeighbors() const { return _nNeighbors; }

    private:
        double euclideanDistance(const std::vector<double> &a, const std::vector<double> &b) const;

        std::vector<std::string> kNearestLabels(const std::vector<double> &query) const;

        std::string majorityVote(const std::vector<std::string> &labels) const;

        void checkIsFitted() const;

        std::size_t _nNeighbors;
        std::vector<std::vector<double> > _xTrain;
        std::vector<std::string> _yTrain;
        std::vector<std::string> _classes;
    };
}
