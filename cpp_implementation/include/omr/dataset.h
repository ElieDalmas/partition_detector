#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace omr {
    // annotation from deepscores_train.json
    struct Annotation {
        std::string imgId; // use like dict keys
        std::array<double, 4> aBbox{0, 0, 0, 0}; // x1, y1, x2, y2
        std::vector<std::string> catId;
        std::string comments;
    };

    struct ImageInfo {
        int id = 0;
        std::string filename;
    };

    // Loader for DeepScoresV2
    class Dataset {
    public:
        void load(const std::string &jsonPath);

        // categories restricted to annotation_set == "deepscores", keyed by their
        // original (string) category id.
        const std::unordered_map<std::string, std::string> &idToName() const { return _idToName; }

        const std::vector<ImageInfo> &images() const { return _images; }

        // nullptr if no image has the filename
        const ImageInfo *findImgByFilename(const std::string &filename) const;

        // Annotations for a given image id (mirrors anns_by_img.get(str(img_id), [])).
        std::vector<Annotation> annsForImg(int imgId) const;

    private:
        std::unordered_map<std::string, std::string> _idToName;
        std::vector<ImageInfo> _images;
        std::unordered_map<std::string, std::vector<Annotation> > _annsByImg;
    };

    // Download tar.gz archive with curl or wget then extract
    bool downloadDataset(const std::string &url, const std::string &destTarGz, const std::string &extractDir);
}
