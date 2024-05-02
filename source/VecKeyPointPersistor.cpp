#include "VecKeyPointPersistor.h"

#include <type_traits>
#include <iostream>

// #include <sys/stat.h>
// #include <sys/mman.h> 
// #include <fcntl.h>
#include <stdio.h>

using namespace std;
using namespace cv;

struct PODKeyPoint {
    float x;
    float y;
    float angle;
    float size;
    int octave;
    float respose;
};

VecKeyPointPersistor::VecKeyPointPersistor()
    : /*file(-1),*/ data(nullptr), length(0) {
}

VecKeyPointPersistor::VecKeyPointPersistor(VecKeyPointPersistor&& o) 
    : file(std::move(o.file)), length(std::move(o.length)), lookup(std::move(o.lookup)), data(std::move(o.data)), keepInMemory(o.keepInMemory) {
    // o.file = -1;
    o.data = nullptr;
}

VecKeyPointPersistor::~VecKeyPointPersistor() {
    close();
}

VecKeyPointPersistor& VecKeyPointPersistor::operator = (VecKeyPointPersistor&& o) {
    file = std::move(o.file); // o.file = -1;
    lookup = std::move(o.lookup);
    data = std::move(o.data); o.data = nullptr;
    length = std::move(o.length); o.length = 0;
    keepInMemory = o.keepInMemory;
    return *this;
}

bool VecKeyPointPersistor::isOpen() {
    // return file != -1;
    return file.is_open();
}

bool VecKeyPointPersistor::exists(const std::string& filename) {
    FILE *file = fopen(filename.c_str(), "r");
    if (file != nullptr) {
        fclose(file);
        return true;
    }
    return false;
}

void VecKeyPointPersistor::close() {
    /*file.release();
    data.release();*/
    // if (file != -1) {
    //     munmap(data, length);
    //     ::close(file);
    // }
    file.close();
    lookup.clear();
}

VecKeyPointPersistor VecKeyPointPersistor::open(const std::string& filename, bool keepInMemory) {
    // int file = ::open(filename.c_str(), O_RDONLY);
    // if (file < 0) {
    //     std::cout << "File does not exist: " << filename << std::endl;
    //     exit(-1);
    //     return VecKeyPointPersistor();
    // }
    
    // struct stat s;
    // int status = fstat(file, &s);
    // if (file < 0) {
    //     std::cout << "Can not read length of file: " << filename << std::endl;
    //     exit(-1);
    //     return VecKeyPointPersistor();
    // }

    /*char* data = nullptr;
    if (keepInMemory) {
        fseek(file, 0, SEEK_END);
        size_t fsize = ftell(file);
        fseek(file, 0, SEEK_SET);

        data = (char*)malloc(fsize + 1);
        size_t ret = fread(data, 1, fsize, file);
        assert(ret > 0);
    }*/
    keepInMemory = true;
    boost::iostreams::mapped_file_source file;
    file.open(filename);

    if (!file.is_open()) {
        std::cout << "File does not exist: " << filename << std::endl;
        exit(-1);
        return VecKeyPointPersistor();
    }

    char* data = (char*)file.data();

    // char* data = (char*)mmap(0, (size_t)s.st_size, PROT_READ, MAP_PRIVATE, file, 0);
    // if (data == MAP_FAILED) {
    //     std::cout << "Can not map file: " << filename << std::endl;
    //     exit(-1);
    //     return VecKeyPointPersistor();
    // }

    VecKeyPointPersistor pers;
    pers.file = file;
    pers.length = file.size(); // s.st_size;
    pers.data = data;
    pers.keepInMemory = keepInMemory;

    int ret;
    size_t descriptorCount = *reinterpret_cast<size_t*>(&data[0]);
    //fseek(file, 0, SEEK_SET);
    /*ret = fread(static_cast<void*>(&descriptorCount), 1, sizeof(descriptorCount), pers.file.get());
    assert(ret == sizeof(descriptorCount));
    std::vector<DescriptorHeader> descriptors;
    descriptors.resize(descriptorCount);
    ret = fread(static_cast<void*>(&descriptors[0]), 1, sizeof(decltype(descriptors)::value_type) * descriptorCount, pers.file.get());
    assert(ret == sizeof(decltype(descriptors)::value_type) * descriptorCount);*/

    for (size_t i = 0; i < descriptorCount; ++i) {
        DescriptorHeader descriptor = *reinterpret_cast<DescriptorHeader*>(&data[sizeof(descriptorCount) + i * sizeof(DescriptorHeader)]);
        pers.lookup[descriptor.frame] = std::make_tuple(descriptor.offset, descriptor.count);
    }

    return pers;
}

bool VecKeyPointPersistor::create(const std::string& filename, const std::vector<std::pair<std::vector<cv::KeyPoint>, uint32_t>>& frames) {
    constexpr size_t KEYPOINT_SIZE = sizeof(((KeyPoint*)0)->pt.x)   +
                                     sizeof(((KeyPoint*)0)->pt.y)   +
                                     sizeof(((KeyPoint*)0)->angle)  +
                                     sizeof(((KeyPoint*)0)->size)   +
                                     sizeof(((KeyPoint*)0)->octave) +
                                     sizeof(((KeyPoint*)0)->response);
    static_assert(sizeof(PODKeyPoint) == KEYPOINT_SIZE, "The stored keypoint must have the same size as the original one!");

    FILE* file = fopen(filename.c_str(), "wb");
    if (file == nullptr) {
        return false;
    }

    size_t offset = sizeof(DescriptorHeader) * frames.size() + sizeof(size_t);
    std::vector<DescriptorHeader> descriptors;
    std::vector<cv::Mat> mats;

    for(size_t i = 0; i < frames.size(); ++i) {
        descriptors.push_back(
            {
                static_cast<uint64_t>(offset),
                static_cast<uint32_t>(frames[i].first.size()),
                static_cast<uint32_t>(frames[i].second)
            }
        );
        offset += sizeof(size_t) + frames[i].first.size() * KEYPOINT_SIZE;
    }

    size_t descriptorHeaders = descriptors.size();
    fwrite(static_cast<void*>(&descriptorHeaders), 1, sizeof(size_t), file);
    fwrite(static_cast<void*>(&descriptors[0]), 1, sizeof(decltype(descriptors)::value_type) * descriptors.size(), file);

    for(size_t i = 0; i < frames.size(); ++i) {
        size_t count = frames[i].first.size();
        fwrite(static_cast<void*>(&count), 1, sizeof(count), file);

        for (size_t j = 0; j < count; ++j) {
            const auto& okp = frames[i].first[j];
            PODKeyPoint kp = {
                okp.pt.x,
                okp.pt.y,
                okp.angle,
                okp.size,
                okp.octave,
                okp.response
            };

            fwrite(static_cast<void*>(&kp), 1, sizeof(kp), file);
        }
    }

    fclose(file);
    return true;
}

bool VecKeyPointPersistor::create(const std::string& filename, const std::vector<std::vector<cv::KeyPoint>>& keypoints, const std::vector<uint32_t>& frameIndices) {
    constexpr size_t KEYPOINT_SIZE = sizeof(((KeyPoint*)0)->pt.x)   +
                                     sizeof(((KeyPoint*)0)->pt.y)   +
                                     sizeof(((KeyPoint*)0)->angle)  +
                                     sizeof(((KeyPoint*)0)->size)   +
                                     sizeof(((KeyPoint*)0)->octave) +
                                     sizeof(((KeyPoint*)0)->response);
    static_assert(sizeof(PODKeyPoint) == KEYPOINT_SIZE, "The stored keypoint must have the same size as the original one!");
    assert(keypoints.size() == frameIndices.size());

    FILE* file = fopen(filename.c_str(), "wb");
    if (file == nullptr) {
        return false;
    }

    size_t offset = sizeof(DescriptorHeader) * keypoints.size() + sizeof(size_t);
    std::vector<DescriptorHeader> descriptors;
    std::vector<cv::Mat> mats;

    for(size_t i = 0; i < keypoints.size(); ++i) {
        descriptors.push_back(
            {
                static_cast<uint64_t>(offset),
                static_cast<uint32_t>(keypoints[i].size()),
                static_cast<uint32_t>(frameIndices[i])
            }
        );
        offset += sizeof(size_t) + keypoints[i].size() * KEYPOINT_SIZE;
    }

    size_t descriptorHeaders = descriptors.size();
    fwrite(static_cast<void*>(&descriptorHeaders), 1, sizeof(size_t), file);
    fwrite(static_cast<void*>(&descriptors[0]), 1, sizeof(decltype(descriptors)::value_type) * descriptors.size(), file);

    for(size_t i = 0; i < keypoints.size(); ++i) {
        size_t count = keypoints[i].size();
        fwrite(static_cast<void*>(&count), 1, sizeof(count), file);

        for (size_t j = 0; j < count; ++j) {
            const auto& okp = keypoints[i][j];
            PODKeyPoint kp = {
                okp.pt.x,
                okp.pt.y,
                okp.angle,
                okp.size,
                okp.octave,
                okp.response
            };

            fwrite(static_cast<void*>(&kp), 1, sizeof(kp), file);
        }
    }

    fclose(file);
    return true;
}

std::vector<cv::KeyPoint> VecKeyPointPersistor::read(const uint64_t index) {
    auto found = lookup.find(index);
    if (found == lookup.end()) {
        return std::vector<cv::KeyPoint>();
    }

    uint64_t offset = std::get<0>(found->second);

    if (!keepInMemory) {
        /*std::lock_guard<std::mutex> lock(mutex);
        if (cache.exist(index)) {
            return cache.get(index);
        }

        fseek(file.get(), offset, SEEK_SET);

        int ret;
        size_t count;
        ret = fread(static_cast<void*>(&count), 1, sizeof(count), file.get());
        assert(ret > 0);

        std::vector<cv::KeyPoint> keypoints;
        keypoints.resize(count);
        for(size_t i = 0; i < count; ++i) {
            PODKeyPoint podkp;
            ret = fread(static_cast<void*>(&podkp), 1, sizeof(podkp), file.get());
            assert(ret > 0);

            keypoints[i] = cv::KeyPoint(podkp.x, podkp.y, podkp.size, podkp.angle, podkp.respose, podkp.octave);
        }
        cache.insert(index, keypoints);
        return keypoints;*/
        assert(false);
        return std::vector<cv::KeyPoint>();
    } else {
        const size_t count = *reinterpret_cast<size_t*>(&data[offset]);
        std::vector<cv::KeyPoint> keypoints;
        keypoints.resize(count);

        offset += sizeof(size_t);
        for (size_t i = 0; i < count; ++i) {
            PODKeyPoint podkp = *reinterpret_cast<PODKeyPoint*>(&data[offset + i * sizeof(PODKeyPoint)]);
            keypoints[i] = cv::KeyPoint(podkp.x, podkp.y, podkp.size, podkp.angle, podkp.respose, podkp.octave);
        }

        return keypoints;
    }
}

bool VecKeyPointPersistor::contains(const uint64_t index) {
    return lookup.find(index) != lookup.end();
}

std::vector<VecKeyPointPersistor::DescriptorInfo> VecKeyPointPersistor::getDescriptorInfo() const {
    std::vector<DescriptorInfo> infos;
    infos.resize(lookup.size());

    size_t counter = 0;
    for (const auto& l : lookup) {
        infos[counter++] = DescriptorInfo { std::get<1>(l.second), l.first };
    }

    std::sort(infos.begin(), infos.end(), [](const auto& a, const auto& b) {
        return a.frame < b.frame;
    });
    return infos;
}