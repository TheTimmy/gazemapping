#include "Catalog.h"

#include <string>
#include <algorithm>
#include <iostream>
#include <boost/filesystem.hpp>

#include "MatPersistor.h"
#include "KeyPointPersistor.h"
#include "VecKeyPointPersistor.h"

Catalog::Catalog(const std::vector<std::string>& videos,
                 const std::vector<std::string>& gazes,
                 const std::vector<uint64_t>& timestamps,
                 const std::string& vocabularyPath,
                 double tracker_rate) 
    : videoPaths(videos), gazes(gazes), timestamps(timestamps), vocabulary(vocabularyPath), tracker_rate(tracker_rate) {

    if (!boost::filesystem::exists(vocabulary))
        boost::filesystem::create_directory(vocabulary);

    for (const auto& filename : videos) {
        boost::filesystem::path path(filename);
        boost::filesystem::path output(vocabulary + "/" + path.stem().string());

        videoNames.push_back(path.stem().string());
        videoStorage.push_back(output.string());
    }
}

void Catalog::convert() {
    const size_t max = 84 * 14000;

    size_t counter = 0;
    for (size_t i = 0; i < videoStorage.size(); ++i) {
        std::vector<std::pair<std::vector<cv::KeyPoint>, uint32_t>> keypoints;

        boost::filesystem::path path(videoStorage[i]); 
        boost::filesystem::directory_iterator end;
        for (boost::filesystem::directory_iterator it(path); it != end; ++it) {
            if (boost::filesystem::is_regular_file(it->path()) && it->path().extension().string() == ".key") {
                if (counter++ % 1024 == 0)
                    std::cout << "\rKeypoint: " << counter << "/" << max << std::flush;

                uint32_t frame = std::stoi(it->path().filename().string());
                std::vector<cv::KeyPoint> kp;
                KeyPointPersistor::restore(it->path().string(), kp);
                keypoints.emplace_back(kp, frame);
            }
        }

        std::sort(keypoints.begin(), keypoints.end(), [](const auto& a, const auto& b) {
            return a.second < b.second;
        });

        VecKeyPointPersistor::create(videoStorage[i] + ".key", keypoints);
    }
}

void Catalog::prefetch(bool openKeypoints, bool openGaze, bool in_memory, std::function<GazePoint(const GazePoint&)> converter) {
    persistors.resize(videoStorage.size());
    gazePersistors.resize(gazes.size());
    keypointPersistors.resize(videoStorage.size());

    std::vector<VideoIndex> indices;
    for (size_t i = 0; i < videoStorage.size(); ++i) {
        if (!persistors[i].isOpen())
            persistors[i] = VecMatPersistor::open(videoStorage[i] + ".vid", in_memory);
        if(openKeypoints && !keypointPersistors[i].isOpen())
            keypointPersistors[i] = VecKeyPointPersistor::open(videoStorage[i] + ".key", in_memory);
        if(openGaze && !gazePersistors[i].isOpen())
            gazePersistors[i] = std::move(GazePointPersistor::open(gazes[i], timestamps[i], tracker_rate, converter));
    }
}

void Catalog::unload() {
    for (size_t i = 0; i < videoStorage.size(); ++i) {
        if (persistors[i].isOpen())
            persistors[i].close();
        if (gazePersistors[i].isOpen())
            gazePersistors[i].close();
        if (keypointPersistors[i].isOpen())
            keypointPersistors[i].close();
    }
}

std::vector<GazePoint> Catalog::getGaze(const VideoIndex& index) {
    return gazePersistors[index.video][index.frame];
}

std::vector<VideoIndex> Catalog::constructDescriptors(size_t subsample, bool in_memory) {
    prefetch(false, false, in_memory);

    std::vector<VideoIndex> indices;
    for (size_t i = 0; i < videoStorage.size(); ++i) {
        const auto frames = persistors[i].getDescriptorInfo();
        for (size_t k = 0; k < frames.size(); k += subsample) {
            const auto& frame = frames[k];
            for (size_t j = 0; j < frame.count; ++j) {
                indices.push_back(
                    VideoIndex(static_cast<uint16_t>(i), frame.frame, static_cast<uint16_t>(j))
                );
            }
        }
    }

    std::sort(indices.begin(), indices.end());
    return indices;
}

std::vector<VideoIndex> Catalog::constructImageDescriptors() {
    prefetch();

    std::vector<VideoIndex> indices;
    for (size_t i = 0; i < videoStorage.size(); ++i) {
        const auto frames = persistors[i].getDescriptorInfo();
        for (const auto& frame : frames) {
            indices.push_back(VideoIndex(static_cast<uint16_t>(i), frame.frame, -1));
        }
    }

    std::sort(indices.begin(), indices.end());
    return indices;
}

std::string& Catalog::getVideoPath(const uint32_t index) {
    return videoPaths[index];
}

std::string& Catalog::getVideoPath(const VideoIndex& index) {
    return videoPaths[index.video];
}

int Catalog::getVideoID(const std::string& name) {
    for (int i = 0; i < videoNames.size(); ++i) {
        if (videoNames[i] == name) {
            return i;
        }
    }
    return -1;
}

std::vector<cv::KeyPoint> Catalog::getImageKeyPoints(const VideoIndex& index) {
    return keypointPersistors[index.video].read(index.frame);
}

size_t Catalog::getVideoLength(size_t videoIndex) const {
    return persistors[videoIndex].getDescriptorInfo().size();
}

std::vector<VecMatPersistor::DescriptorInfo> Catalog::getVideoInfo(size_t videoIndex) const {
    return persistors[videoIndex].getDescriptorInfo();
}

cv::Mat Catalog::getImageDescriptors(const VideoIndex& index) {
    return persistors[index.video].read(index.frame);
}

cv::Mat Catalog::getDescriptor(const VideoIndex& index) {
    cv::Mat allDesc = getImageDescriptors(index);
    return allDesc.row(index.descriptor);
}
