#pragma once

#include <tuple>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>

#include <opencv2/core.hpp>
#include <boost/filesystem.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

#include "VideoIndex.h"

/**
 * This class is used to persist a Matrix from OpenCV to disk
 * OpenCV built-in persistence to XML/YML has to much overhead and is not fast enough
 */
class VecKeyPointPersistor {
public:
    struct DescriptorInfo {
        uint32_t count;
        uint32_t frame;
    };

    VecKeyPointPersistor();
    VecKeyPointPersistor(const VecKeyPointPersistor&) = delete;
    VecKeyPointPersistor(VecKeyPointPersistor&& o);
    ~VecKeyPointPersistor();

    VecKeyPointPersistor& operator = (VecKeyPointPersistor&& o);
    VecKeyPointPersistor& operator = (const VecKeyPointPersistor&) = delete;

    bool isOpen();
    void close();
    static bool create(const std::string& filename, const std::vector<std::pair<std::vector<cv::KeyPoint>, uint32_t>>& mat);
    static bool create(const std::string& filename, const std::vector<std::vector<cv::KeyPoint>>& keypoints, const std::vector<uint32_t>& frames);
    static VecKeyPointPersistor open(const std::string& filename, bool keepInMemory=false);
    
    void read(std::vector<cv::Mat> &mat);
    std::vector<cv::KeyPoint> read(const uint64_t index);
    bool contains(const uint64_t index);

    std::vector<VecKeyPointPersistor::DescriptorInfo> getDescriptorInfo() const;

    static bool exists(const std::string& filename);

private:
    void readHeader();

    struct DescriptorHeader {
        uint64_t offset;
        uint32_t count;
        uint32_t frame;
    };

    //std::unique_ptr<FILE, int (*)(FILE*)> file;
    //std::unique_ptr<char[], void (*)(char*)> data;
    char* data;    
    std::unordered_map<uint32_t, std::tuple<uint64_t, uint32_t>> lookup;
    std::mutex mutex;
    size_t length;
    // int file;
    boost::iostreams::mapped_file_source file;
    bool keepInMemory;
};