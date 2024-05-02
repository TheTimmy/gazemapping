#pragma once

#include <vector>
#include <string>
#include <opencv2/core.hpp>

#include "VideoIndex.h"
#include "VecMatPersistor.h"
#include "VecKeyPointPersistor.h"
#include "GazePointPersistor.h"

class Catalog {
public:
    Catalog(const std::vector<std::string>& videos,
            const std::vector<std::string>& gaze,
            const std::vector<uint64_t>& timestamps,
            const std::string& vocabularyPath,
            double tracker_rate);

    void convert();

    std::string& getVideoPath(const uint32_t index);
    std::string& getVideoPath(const VideoIndex& index);

    const std::string& getVideoStore(const int videoID);
    int getVideoID(const std::string& name);

    std::vector<cv::KeyPoint> getImageKeyPoints(const VideoIndex& index);

    std::vector<GazePoint> getGaze(const VideoIndex& index);
    cv::Mat getImageDescriptors(const VideoIndex& index);
    cv::Mat getDescriptor(const VideoIndex& index);

    void prefetch(bool openKeypoints=false, bool openGaze=false, bool in_memory=false, std::function<GazePoint(const GazePoint&)> converter=nullptr);
    void unload();

    // std::vector<VecKeyPointPersistor::DescriptorInfo> getVideoInfo(size_t videoIndex) const;
    size_t getVideoLength(size_t videoIndex) const;
    std::vector<VecMatPersistor::DescriptorInfo> getVideoInfo(size_t videoIndex) const;
    std::vector<VideoIndex> constructDescriptors(size_t subsample=1, bool in_memory=false);
    std::vector<VideoIndex> constructImageDescriptors();

    const std::vector<std::string>& getVideoPaths() const { return videoPaths; }
    const std::vector<std::string>& getVideoStorage() const { return videoStorage; }
    const std::vector<std::string>& getVideoNames() const { return videoNames; }
    const std::string& getDatabase() const { return vocabulary; }

    const std::vector<size_t> getVideoIndices() const {
        std::vector<size_t> indices;
        for (size_t i = 0; i < videoPaths.size(); ++i) {
            indices.push_back(i);
        }
        return indices;
    }

private:
    std::vector<VecMatPersistor> persistors;
    std::vector<VecKeyPointPersistor> keypointPersistors;
    std::vector<GazePointPersistor> gazePersistors;

    std::vector<std::string> videoPaths;
    std::vector<std::string> videoStorage;
    std::vector<std::string> videoNames;
    std::vector<std::string> gazes;
    std::vector<uint64_t> timestamps;

    std::string vocabulary;
    double tracker_rate;
};