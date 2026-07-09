#include "omr/dataset.h"

#include "json.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace omr {
    namespace {
        using nlohmann::json;

        // normalize json Id
        std::string jsonToKeyString(const json &j) {
            if (j.is_string()) {
                return j.get<std::string>();
            }
            return j.dump();
        }

        // parse json number
        int jsonToInt(const json &j) {
            if (j.is_number_integer()) {
                return j.get<int>();
            }
            if (j.is_number_float()) {
                return static_cast<int>(j.get<double>());
            }
            return std::stoi(j.get<std::string>());
        }
    }

    void Dataset::load(const std::string &jsonPath) {
        std::ifstream f(jsonPath);
        if (!f) {
            throw std::runtime_error("Cannot open dataset json: " + jsonPath);
        }

        json data;
        f >> data;

        _idToName.clear();
        // map id to name
        for (auto &[key, val]: data.at("categories").items()) {
            std::string annSet = val.value("annotation_set", std::string());
            if (annSet == "deepscores") {
                _idToName[key] = val.at("name").get<std::string>();
            }
        }

        _images.clear();
        // list (id, img_path)
        for (auto &imgJson: data.at("images")) {
            ImageInfo info;
            info.id = jsonToInt(imgJson.at("id"));
            info.filename = imgJson.at("filename").get<std::string>();
            _images.push_back(std::move(info));
        }

        _annsByImg.clear();
        for (auto &[annId, annJson]: data.at("annotations").items()) {
            Annotation ann;
            ann.imgId = jsonToKeyString(annJson.at("img_id"));

            // bbox
            const auto &bboxArr = annJson.at("a_bbox");
            for (int i = 0; i < 4; ++i) {
                ann.aBbox[static_cast<std::size_t>(i)] = bboxArr.at(static_cast<std::size_t>(i)).get<double>();
            }

            // category id
            for (const auto &c: annJson.at("cat_id")) {
                ann.catId.push_back(jsonToKeyString(c));
            }

            ann.comments = annJson.value("comments", std::string());

            _annsByImg[ann.imgId].push_back(std::move(ann));
        }
    }

    const ImageInfo *Dataset::findImgByFilename(const std::string &filename) const {
        for (const auto &info: _images) {
            if (info.filename == filename) {
                return &info;
            }
        }
        return nullptr;
    }

    std::vector<Annotation> Dataset::annsForImg(int imgId) const {
        auto it = _annsByImg.find(std::to_string(imgId));
        if (it == _annsByImg.end()) {
            return {};
        }
        return it->second;
    }

    bool downloadDataset(const std::string &url, const std::string &destTarGz, const std::string &extractDir) {
        std::error_code ec;
        std::filesystem::path destPath(destTarGz);
        if (destPath.has_parent_path()) {
            std::filesystem::create_directories(destPath.parent_path(), ec);
        }
        std::filesystem::create_directories(extractDir, ec);

        std::cout << "Downloading dataset from " << url << " ..." << std::endl;
        std::string curlCmd = "curl -L --fail -o \"" + destTarGz + "\" \"" + url + "\"";
        int res = std::system(curlCmd.c_str());

        if (res != 0) {
            std::cout << "curl failed (or unavailable), trying wget ..." << std::endl;
            std::string wgetCmd = "wget -O \"" + destTarGz + "\" \"" + url + "\"";
            res = std::system(wgetCmd.c_str());
        }

        if (res != 0) {
            std::cerr << "Download failed." << std::endl;
            return false;
        }

        std::cout << "Extracting to " << extractDir << " ..." << std::endl;
        std::string tarCmd = "tar -xzf \"" + destTarGz + "\" -C \"" + extractDir + "\"";
        res = std::system(tarCmd.c_str());
        if (res != 0) {
            std::cerr << "Extraction failed." << std::endl;
            return false;
        }

        return true;
    }
}