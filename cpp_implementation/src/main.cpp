#include "omr/classifier.h"
#include "omr/dataset.h"
#include "omr/evaluation.h"
#include "omr/features.h"
#include "omr/otsu.h"
#include "omr/pitch.h"
#include "omr/preprocessing.h"
#include "omr/segmentation.h"
#include "omr/staff.h"
#include "omr/box.h"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

// Arguments parsing + load image fun
namespace {
    using Clock = std::chrono::steady_clock;

    double elapsedMs(Clock::time_point start) {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }

    // Same 8 training images as python
    const std::vector<std::string> DEFAULT_TRAIN_FILENAMES = {
        "lg-233786100286899765-aug-gutenberg1939--page-2.png",
        "lg-797958438447858728-aug-lilyjazz--page-1.png",
        "lg-827327016295735829-aug-beethoven--page-1.png",
        "lg-138696691-aug-gutenberg1939--page-1.png",
        "lg-523155381240242384-aug-lilyjazz--page-41.png",
        "lg-214320686-aug-lilyjazz--page-3.png",
        "lg-43865734-aug-beethoven--page-2.png",
        "lg-359041762321484926-aug-emmentaler-.png",
    };

    const std::string DEFAULT_SAMPLE_FILENAME = "lg-214320686-aug-lilyjazz--page-3.png";
    const std::string DEFAULT_DATASET_URL = "https://zenodo.org/records/4012193/files/ds2_dense.tar.gz";

    struct Args {
        std::string dataDir = "data/ds2_dense";
        std::string sampleFilename = DEFAULT_SAMPLE_FILENAME;
        std::vector<std::string> trainFilenames = DEFAULT_TRAIN_FILENAMES;
        std::size_t k = 5;
        bool download = false;
        std::string datasetUrl = DEFAULT_DATASET_URL;
        std::string dumpDir; // empty = don't dump images
    };

    std::vector<std::string> splitCsv(const std::string &s) {
        std::vector<std::string> out;
        std::string cur;
        for (char c: s) {
            if (c == ',') {
                if (!cur.empty()) out.push_back(cur);
                cur.clear();
            } else {
                cur.push_back(c);
            }
        }
        if (!cur.empty()) out.push_back(cur);
        return out;
    }

    Args parseArgs(int argc, char **argv) {
        Args args;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto nextVal = [&](const char *flagName) -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error(std::string("Missing value for ") + flagName);
                }
                return argv[++i];
            };

            if (arg == "--data-dir") {
                args.dataDir = nextVal("--data-dir");
            } else if (arg == "--sample") {
                args.sampleFilename = nextVal("--sample");
            } else if (arg == "--train") {
                args.trainFilenames = splitCsv(nextVal("--train"));
            } else if (arg == "--k") {
                args.k = std::stoul(nextVal("--k"));
            } else if (arg == "--download") {
                args.download = true;
            } else if (arg == "--dump-images") {
                args.dumpDir = nextVal("--dump-images");
            } else if (arg == "--help" || arg == "-h") {
                std::cout <<
                        "Usage: omr_pipeline [options]\n"
                        "  --data-dir <dir> : Dataset root (default=data/ds2_dense)\n"
                        "  --sample <filename> : Test image filename\n"
                        "  --train <filename1,...> : Training image filenames\n"
                        "  --k <int> : kNN neighbors (default=5)\n"
                        "  --download : Download + extract the dataset first\n"
                        "  --dump-images <dir> : Write overlay png for visual checks\n";
                std::exit(0);
            } else {
                std::cerr << "Unknown argument: " << arg << " (\\0_0/ need --help)\n";
                std::exit(1);
            }
        }
        return args;
    }

    // {bgr, gray, bin}
    struct LoadedImage {
        cv::Mat bgr;
        cv::Mat gray;
        cv::Mat bin;
    };

    // load cv img, apply grayscale + otsu binarize
    LoadedImage loadAndBinarize(const std::string &path) {
        LoadedImage out;
        out.bgr = cv::imread(path, cv::IMREAD_COLOR);
        if (out.bgr.empty()) {
            throw std::runtime_error("Could not read image: " + path);
        }
        cv::cvtColor(out.bgr, out.gray, cv::COLOR_BGR2GRAY);
        auto [thr, bin] = omr::OtsuThreshold::apply(out.gray);
        (void) thr;
        out.bin = bin;
        return out;
    }
}

int main(int argc, char **argv) {
    Args args;
    try {
        args = parseArgs(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << "Argument error: " << e.what() << std::endl;
        return 1;
    }

    auto tStart = Clock::now();

    try {
        // Download dataset
        if (args.download) {
            std::filesystem::path dataDirPath(args.dataDir);
            std::string extractDir = dataDirPath.has_parent_path()
                                         ? dataDirPath.parent_path().string()
                                         : ".";
            std::string destTarGz = extractDir + "/ds2_dense.tar.gz";
            if (!omr::downloadDataset(args.datasetUrl, destTarGz, extractDir)) {
                std::cerr << "Dataset download/extraction failed." << std::endl;
                return 1;
            }
        }

        // Dataset
        auto t = Clock::now();
        omr::Dataset dataset;
        dataset.load(args.dataDir + "/deepscores_train.json");
        std::cout << "[dataset] loaded " << dataset.images().size() << " images in "
                << elapsedMs(t) << " ms" << std::endl;

        const auto *sampleInfo = dataset.findImgByFilename(args.sampleFilename);
        if (sampleInfo == nullptr) {
            std::cerr << "Sample image not found in dataset: " << args.sampleFilename << std::endl;
            return 1;
        }
        auto sampleAnns = dataset.annsForImg(sampleInfo->id);
        const auto &idToName = dataset.idToName();

        // Load + binarize
        t = Clock::now();
        LoadedImage sample = loadAndBinarize(args.dataDir + "/images/" + sampleInfo->filename);
        int hImg = sample.bin.rows;
        int wImg = sample.bin.cols;
        std::cout << "[otsu] " << wImg << "x" << hImg << " in " << elapsedMs(t) << " ms" << std::endl;

        // Ground truth noteheads
        std::vector<omr::Box> gtNoteheads;
        for (const auto &ann: sampleAnns) {
            std::string cat = ann.catId.empty() ? "?" : idToName.count(ann.catId[0]) ? idToName.at(ann.catId[0]) : "?";
            if (omr::NOTEHEAD_CATS.count(cat)) {
                gtNoteheads.push_back(omr::Box{
                    static_cast<int>(ann.aBbox[0]), static_cast<int>(ann.aBbox[1]),
                    static_cast<int>(ann.aBbox[2]), static_cast<int>(ann.aBbox[3])
                });
            }
        }
        if (gtNoteheads.empty()) {
            std::cerr << "No GT noteheads found for sample image." << std::endl;
            return 1;
        }
        double avgW = 0.0, avgH = 0.0;
        for (const auto &b: gtNoteheads) {
            avgW += b.width();
            avgH += b.height();
        }
        avgW /= gtNoteheads.size();
        avgH /= gtNoteheads.size();
        std::cout << "[gt] " << gtNoteheads.size() << " noteheads, avg size " << avgW << " x " << avgH << " px" <<
                std::endl;

        // Preprocessing
        t = Clock::now();
        cv::Mat imgClean = omr::removeStaffBarsAndStems(sample.bin, avgH);
        std::cout << "[preprocessing] done in " << elapsedMs(t) << " ms" << std::endl;

        // Segmentation
        t = Clock::now();
        auto detectedNoteheads = omr::detectNoteheads(imgClean, avgW, avgH);
        std::cout << "[segmentation] " << detectedNoteheads.size() << " candidates in " << elapsedMs(t) << " ms" <<
                std::endl;

        // Detection evaluation
        auto detEval = omr::evaluate(detectedNoteheads, gtNoteheads);
        std::cout << "[detection] TP=" << detEval.tp << " FP=" << detEval.fp << " FN=" << detEval.fn
                << " P=" << detEval.precision << " R=" << detEval.recall << " F1=" << detEval.f1 << std::endl;

        // Training
        t = Clock::now();
        std::vector<std::vector<double> > xAll;
        std::vector<std::string> yL1All;
        std::vector<std::optional<std::string> > yL2All;

        for (const auto &fname: args.trainFilenames) {
            const auto *info = dataset.findImgByFilename(fname);
            if (info == nullptr) continue;
            LoadedImage tr = loadAndBinarize(args.dataDir + "/images/" + fname);

            auto anns = dataset.annsForImg(info->id);
            auto trainData = omr::buildTrainingData(anns, idToName, tr.bin);

            xAll.insert(xAll.end(), trainData.X.begin(), trainData.X.end());
            yL1All.insert(yL1All.end(), trainData.yL1.begin(), trainData.yL1.end());
            yL2All.insert(yL2All.end(), trainData.yL2.begin(), trainData.yL2.end());
        }
        omr::SymbolClassifier clf(args.k);
        clf.fit(xAll, yL1All, yL2All);
        std::cout << "[training] " << xAll.size() << " samples from " << args.trainFilenames.size() << " images in "
                << elapsedMs(t) << " ms" << std::endl;

        // Classification
        t = Clock::now();
        std::vector<std::vector<double> > xTest;
        xTest.reserve(detectedNoteheads.size());
        for (const auto &box: detectedNoteheads) {
            cv::Rect rect(box.x1, box.y1, box.width(), box.height());
            rect &= cv::Rect(0, 0, sample.bin.cols, sample.bin.rows);
            cv::Mat symbol = sample.bin(rect);
            xTest.push_back(omr::extractFeatures(symbol));
        }
        auto preds = clf.predict(xTest);
        std::cout << "[classification] " << preds.size() << " blobs in " << elapsedMs(t) << " ms" << std::endl;

        // Level 2 accuracy
        // match detected boxes with near GT ann
        int goodL2 = 0, totalL2 = 0;
        for (std::size_t i = 0; i < detectedNoteheads.size(); ++i) {
            double cx = detectedNoteheads[i].centerX();
            double cy = detectedNoteheads[i].centerY();
            for (const auto &ann: sampleAnns) {
                double annCx = (ann.aBbox[0] + ann.aBbox[2]) / 2.0;
                double annCy = (ann.aBbox[1] + ann.aBbox[3]) / 2.0;

                if (std::abs(cx - annCx) < avgW && std::abs(cy - annCy) < avgH) {
                    std::string cat = ann.catId.empty()
                                          ? "?"
                                          : idToName.count(ann.catId[0])
                                                ? idToName.at(ann.catId[0])
                                                : "?";

                    auto gtL2 = omr::level2Label(cat);
                    auto predL2 = preds[i].second;
                    if (gtL2.has_value() && predL2.has_value()) {
                        totalL2++;
                        if (*gtL2 == *predL2) goodL2++;
                    }
                    break;
                }
            }
        }
        if (totalL2 > 0) {
            std::cout << "[level2 accuracy] " << (static_cast<double>(goodL2) / totalL2)
                    << " (" << goodL2 << "/" << totalL2 << ")" << std::endl;
        }

        // F1 @ IoU>=0.5
        std::vector<std::pair<std::string, const std::unordered_set<std::string> *> > classes = {
            {"black", &omr::NOTEHEAD_BLACK},
            {"half", &omr::NOTEHEAD_HALF},
            {"whole", &omr::NOTEHEAD_WHOLE}
        };

        std::vector<double> apValues;
        for (const auto &[className, catSet]: classes) {
            std::vector<omr::Box> gtC, detC;
            for (const auto &ann: sampleAnns) {
                std::string cat = ann.catId.empty()
                                      ? "?"
                                      : idToName.count(ann.catId[0])
                                            ? idToName.at(ann.catId[0])
                                            : "?";

                if (catSet->count(cat)) {
                    gtC.push_back(omr::Box{
                        static_cast<int>(ann.aBbox[0]), static_cast<int>(ann.aBbox[1]),
                        static_cast<int>(ann.aBbox[2]), static_cast<int>(ann.aBbox[3])
                    });
                }
            }
            for (std::size_t i = 0; i < detectedNoteheads.size(); ++i) {
                if (preds[i].second.has_value() && *preds[i].second == className) {
                    detC.push_back(detectedNoteheads[i]);
                }
            }
            double ap = omr::apAt05(detC, gtC);
            apValues.push_back(ap);
            std::cout << "[F1@IoU0.5] " << className << ": " << ap << std::endl;
        }
        if (!apValues.empty()) {
            double mAp = std::accumulate(apValues.begin(), apValues.end(), 0.0) / apValues.size();
            std::cout << "[mAP] " << mAp << std::endl;
        }

        // Staff line detection + grouping
        t = Clock::now();
        auto staffLineYs = omr::detectStaffLines(sample.bin);
        auto staves = omr::groupStaves(staffLineYs);
        std::cout << "[staff] " << staffLineYs.size() << " lines, " << staves.size() << " staves in "
                << elapsedMs(t) << " ms" << std::endl;

        // Pitch accuracy
        // default: Sol/G key
        std::vector<std::tuple<omr::Box, int> > gtPitches;
        for (const auto &ann: sampleAnns) {
            std::string cat = ann.catId.empty() ? "?" : idToName.count(ann.catId[0]) ? idToName.at(ann.catId[0]) : "?";
            if (omr::NOTEHEAD_CATS.count(cat)) {
                auto rel = omr::parseRelPosition(ann.comments);
                if (rel.has_value()) {
                    omr::Box box{
                        static_cast<int>(ann.aBbox[0]), static_cast<int>(ann.aBbox[1]),
                        static_cast<int>(ann.aBbox[2]), static_cast<int>(ann.aBbox[3])
                    };
                    gtPitches.emplace_back(box, *rel);
                }
            }
        }

        int goodPitch = 0, totalPitch = 0;
        if (!staves.empty()) {
            for (const auto &box: detectedNoteheads) {
                double cx = box.centerX();
                double cy = omr::getCyFromPixels(box, sample.bin);

                const std::vector<int> *bestStaff = nullptr;
                double bestDist = std::numeric_limits<double>::infinity();
                for (const auto &staff: staves) {
                    for (int y: staff) {
                        double d = std::abs(cy - y);
                        if (d < bestDist) {
                            bestDist = d;
                            bestStaff = &staff;
                        }
                    }
                }
                if (bestStaff == nullptr) continue;
                int predRel = omr::relPositionFromY(cy, *bestStaff);

                std::optional<int> gtRel;
                for (const auto &[gtBox, rel]: gtPitches) {
                    if (gtBox.x1 <= cx && cx <= gtBox.x2 && gtBox.y1 <= cy && cy <= gtBox.y2) {
                        gtRel = rel;
                        break;
                    }
                }
                if (gtRel.has_value()) {
                    totalPitch++;
                    if (predRel == *gtRel) goodPitch++;
                }
            }
        }
        if (totalPitch > 0) {
            std::cout << "[pitch accuracy] " << (static_cast<double>(goodPitch) / totalPitch) << " ("
                    << goodPitch << "/" << totalPitch << ")" << std::endl;
        }

        // Optional overlay visual
        if (!args.dumpDir.empty()) {
            std::filesystem::create_directories(args.dumpDir);
            cv::Mat segVis = sample.bgr.clone();
            for (const auto &b: detectedNoteheads) {
                cv::rectangle(segVis, {b.x1, b.y1}, {b.x2, b.y2}, cv::Scalar(0, 200, 0), 1);
            }
            cv::imwrite(args.dumpDir + "/segmentation.png", segVis);

            cv::Mat staffVis = sample.bgr.clone();
            for (int y: staffLineYs) {
                cv::line(staffVis, {0, y}, {wImg, y}, cv::Scalar(255, 0, 0), 2);
            }
            cv::imwrite(args.dumpDir + "/staff.png", staffVis);
        }

        std::cout << "[total] " << elapsedMs(tStart) << " ms" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
