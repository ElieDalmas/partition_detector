#include "omr/classifier.h"
#include "omr/features.h"

namespace omr {
    // level 1
    const std::unordered_set<std::string> NOTEHEAD_CATS = {
        "noteheadBlackOnLine",
        "noteheadBlackInSpace",
        "noteheadWholeOnLine",
        "noteheadWholeInSpace",
        "noteheadHalfOnLine",
        "noteheadHalfInSpace",
    };

    const std::unordered_set<std::string> REST_CATS = {
        "restWhole",
        "restHalf",
        "restQuarter",
        "rest8th",
        "rest16th",
        "rest32nd",
    };

    const std::unordered_set<std::string> CLEF_CATS = {"clefG", "clefF", "clefC"};

    // level 2
    const std::unordered_set<std::string> NOTEHEAD_BLACK = {"noteheadBlackOnLine", "noteheadBlackInSpace"};
    const std::unordered_set<std::string> NOTEHEAD_HALF = {"noteheadHalfOnLine", "noteheadHalfInSpace"};
    const std::unordered_set<std::string> NOTEHEAD_WHOLE = {"noteheadWholeOnLine", "noteheadWholeInSpace"};

    std::string level1Label(const std::string &catName) {
        if (NOTEHEAD_CATS.count(catName)) return "notehead";
        if (REST_CATS.count(catName)) return "rest";
        if (CLEF_CATS.count(catName)) return "clef";
        return "other";
    }

    std::optional<std::string> level2Label(const std::string &catName) {
        if (NOTEHEAD_BLACK.count(catName)) return "black";
        if (NOTEHEAD_HALF.count(catName)) return "half";
        if (NOTEHEAD_WHOLE.count(catName)) return "whole";
        return std::nullopt;
    }

    TrainingData buildTrainingData(
        const std::vector<Annotation> &anns,
        const std::unordered_map<std::string, std::string> &idToName,
        const cv::Mat &imgBin) {
        TrainingData out;

        for (const auto &ann: anns) {
            // determine categories (l1, l2)
            std::string cat = "?";
            if (!ann.catId.empty()) {
                auto it = idToName.find(ann.catId[0]);
                if (it != idToName.end()) {
                    cat = it->second;
                }
            }

            std::string l1 = level1Label(cat);
            std::optional<std::string> l2 = level2Label(cat);

            // bbox from annotations
            int x1 = static_cast<int>(ann.aBbox[0]);
            int y1 = static_cast<int>(ann.aBbox[1]);
            int x2 = static_cast<int>(ann.aBbox[2]);
            int y2 = static_cast<int>(ann.aBbox[3]);
            int w = x2 - x1;
            int h = y2 - y1;
            if (w < 4 || h < 4 || w > 200 || h > 200) {
                continue;
            }

            // extract symbol from binary img
            cv::Rect symbolBox(x1, y1, w, h);
            cv::Rect imgRect(0, 0, imgBin.cols, imgBin.rows);
            symbolBox &= imgRect;
            if (symbolBox.width <= 0 || symbolBox.height <= 0) {
                continue;
            }
            cv::Mat symbol = imgBin(symbolBox);

            out.X.push_back(extractFeatures(symbol));
            out.yL1.push_back(l1);
            out.yL2.push_back(l2);
        }

        return out;
    }

    SymbolClassifier::SymbolClassifier(std::size_t k) : _knnL1(k), _knnL2(k) {
    }

    void SymbolClassifier::fit(const std::vector<std::vector<double> > &X,
                               const std::vector<std::string> &yL1,
                               const std::vector<std::optional<std::string> > &yL2) {
        _knnL1.fit(X, yL1);

        // keep only notehead
        std::vector<std::vector<double> > xNh;
        std::vector<std::string> yNh;
        for (std::size_t i = 0; i < yL1.size(); ++i) {
            if (yL1[i] == "notehead" && yL2[i].has_value()) {
                xNh.push_back(X[i]);
                yNh.push_back(*yL2[i]);
            }
        }
        _knnL2.fit(xNh, yNh);
    }

    std::vector<std::pair<std::string, std::optional<std::string> > >
    SymbolClassifier::predict(const std::vector<std::vector<double> > &X) const {
        auto l1Preds = _knnL1.predict(X);

        std::vector<std::pair<std::string, std::optional<std::string> > > res;
        res.reserve(X.size());

        for (std::size_t i = 0; i < l1Preds.size(); ++i) {
            std::optional<std::string> l2;
            if (l1Preds[i] == "notehead") {
                auto l2Pred = _knnL2.predict({X[i]});
                l2 = l2Pred[0];
            }
            res.emplace_back(l1Preds[i], l2);
        }

        return res;
    }
}
