#include "VecMatPersistor.h"
#include <mutex>
#include <iostream>

// #include <sys/stat.h>
// #include <sys/mman.h> 
// #include <fcntl.h>
#include <stdio.h>

using namespace std;
using namespace cv;

VecMatPersistor::VecMatPersistor() 
    : /*file(-1),*/ data(nullptr), length(0) {
}

VecMatPersistor::VecMatPersistor(VecMatPersistor&& o) 
    : file(std::move(o.file)), length(std::move(o.length)), data(std::move(o.data)), lookup(std::move(o.lookup)), keepInMemory(o.keepInMemory) {
    // o.file = -1;
    o.data = nullptr;
}

VecMatPersistor::~VecMatPersistor() {
    close();
}

VecMatPersistor& VecMatPersistor::operator = (VecMatPersistor&& o) {
    file = o.file; // o.file = -1;
    data = std::move(o.data); o.data = nullptr;
    lookup = std::move(o.lookup);
    length = std::move(o.length);
    keepInMemory = o.keepInMemory;
    return *this;
}

bool VecMatPersistor::isOpen() {
    // return file != -1;
    return file.is_open();
}

bool VecMatPersistor::exists(const std::string& filename) {
    FILE *file = fopen(filename.c_str(), "r");
    if (file != nullptr) {
        fclose(file);
        return true;
    }
    return false;
}

void VecMatPersistor::close() {
    //file.release();
    // if (file != -1) {
    //     munmap(data, length);
    //     ::close(file);
    // }
    file.close();
    lookup.clear();
}

VecMatPersistor VecMatPersistor::open(const std::string& filename, bool keepInMemory) {
    //FILE* file = fopen(filename.c_str(), "rb");
    // int file = ::open(filename.c_str(), O_RDONLY);
    // if (file < 0) {
    //     std::cout << "File does not exist: " << filename << std::endl;
    //     exit(-1);
    //     return VecMatPersistor();
    // }

    // struct stat s;
    // int status = fstat(file, &s);
    // if (file < 0) {
    //     std::cout << "Can not read length of file: " << filename << std::endl;
    //     exit(-1);
    //     return VecMatPersistor();
    // }


    /*char* data = nullptr;
    if (keepInMemory) {
        fseek(file, 0, SEEK_END);
        size_t fsize = ftell(file);
        fseek(file, 0, SEEK_SET);

        data = (char*)malloc(fsize + 1);
        size_t ret = fread(data, 1, fsize, file);
        assert(ret == fsize);
    } else {

    }*/
    keepInMemory = true;
    boost::iostreams::mapped_file_source file;
    file.open(filename);

    if (!file.is_open()) {
        std::cout << "File does not exist: " << filename << std::endl;
        exit(-1);
        return VecMatPersistor();
    }

    char* data = (char*)file.data();
    
    // char* data = (char*)mmap(0, (size_t)s.st_size, PROT_READ, MAP_PRIVATE, file, 0);
    // if (data == MAP_FAILED) {
    //     std::cout << "Can not map file: " << filename << std::endl;
    //     exit(-1);
    //     return VecMatPersistor();
    // }

    VecMatPersistor pers;
    pers.data = data;
    pers.length = file.size(); // s.st_size;
    pers.file = file;
    pers.keepInMemory = keepInMemory;

    int ret;
    size_t descriptorCount = *reinterpret_cast<size_t*>(&data[0]);

    //fseek(file, 0, SEEK_SET);
    //ret = fread(static_cast<void*>(&descriptorCount), 1, sizeof(descriptorCount), pers.file.get());
    //assert(ret == sizeof(descriptorCount));

    /*std::vector<DescriptorHeader> descriptors;
    descriptors.resize(descriptorCount);
    ret = fread(static_cast<void*>(&descriptors[0]), 1, sizeof(decltype(descriptors)::value_type) * descriptorCount, pers.file.get());
    if (ret != sizeof(decltype(descriptors)::value_type) * descriptorCount) {
        std::cout << "Error reading file: " << filename << std::endl;
    }
    assert(ret == sizeof(decltype(descriptors)::value_type) * descriptorCount);*/

    for (size_t i = 0; i < descriptorCount; ++i) {
        DescriptorHeader descriptor = *reinterpret_cast<DescriptorHeader*>(&data[sizeof(descriptorCount) + i * sizeof(DescriptorHeader)]);
        pers.lookup[descriptor.frame] = std::make_tuple(descriptor.offset, descriptor.count);
    }

    return pers;
}

bool VecMatPersistor::create(const std::string& filename, const std::vector<std::pair<Mat, uint32_t>>& frames) {
    FILE* file = fopen(filename.c_str(), "wb");
    if (file == nullptr) {
        return false;
    }

    size_t offset = sizeof(DescriptorHeader) * frames.size() + sizeof(size_t);
    std::vector<DescriptorHeader> descriptors;
    for(size_t i = 0; i < frames.size(); ++i) {
        descriptors.push_back(
            {
                static_cast<uint64_t>(offset),
                static_cast<uint32_t>(frames[i].first.rows),
                static_cast<uint32_t>(frames[i].second)
            }
        );
        offset += sizeof(MatHeader) + frames[i].first.rows * frames[i].first.cols * frames[i].first.elemSize();
    }

    size_t descriptorHeaders = descriptors.size();
    fwrite(static_cast<void*>(&descriptorHeaders), 1, sizeof(size_t), file);
    fwrite(static_cast<void*>(&descriptors[0]), 1, sizeof(decltype(descriptors)::value_type) * descriptors.size(), file);

    for(size_t i = 0; i < frames.size(); ++i) {
        long bytes = frames[i].first.rows * frames[i].first.cols * frames[i].first.elemSize();
        MatHeader header = {frames[i].first.cols, frames[i].first.rows, frames[i].first.type() };
        fwrite(static_cast<void*>(&header), 1, sizeof(header), file);
        fwrite(static_cast<void*>(frames[i].first.data), 1, bytes, file);
    }

    fclose(file);
    return true;
}

bool VecMatPersistor::create(const std::string& filename, const std::vector<Mat>& mats, const std::vector<uint32_t>& frameIndices) {
    assert(mats.size() == frameIndices.size());

    FILE* file = fopen(filename.c_str(), "wb");
    if (file == nullptr) {
        return false;
    }

    size_t offset = sizeof(DescriptorHeader) * mats.size() + sizeof(size_t);
    std::vector<DescriptorHeader> descriptors;
    for(size_t i = 0; i < mats.size(); ++i) {
        descriptors.push_back(
            {
                static_cast<uint64_t>(offset),
                static_cast<uint32_t>(mats[i].rows),
                static_cast<uint32_t>(frameIndices[i])
            }
        );
        offset += sizeof(MatHeader) + mats[i].rows * mats[i].cols * mats[i].elemSize();
    }

    size_t descriptorHeaders = descriptors.size();
    fwrite(static_cast<void*>(&descriptorHeaders), 1, sizeof(size_t), file);
    fwrite(static_cast<void*>(&descriptors[0]), 1, sizeof(decltype(descriptors)::value_type) * descriptors.size(), file);

    for(size_t i = 0; i < mats.size(); ++i) {
        long bytes = mats[i].rows * mats[i].cols * mats[i].elemSize();
        MatHeader header = {mats[i].cols, mats[i].rows, mats[i].type() };
        fwrite(static_cast<void*>(&header), 1, sizeof(header), file);
        fwrite(static_cast<void*>(mats[i].data), 1, bytes, file);
    }

    fclose(file);
    return true;
}

cv::Mat VecMatPersistor::read(const uint64_t index) {
    auto found = lookup.find(index);
    if (found == lookup.end()) {
        return cv::Mat();
    }

    uint64_t offset = std::get<0>(found->second);
    if (!keepInMemory) {
        std::lock_guard<std::mutex> lock(mutex);
        /*fseek(file.get(), offset, SEEK_SET);
        
        int ret;
        MatHeader header;
        ret = fread(static_cast<void*>(&header), 1, sizeof(header), file.get());
        assert(ret > 0);
        cv::Mat mat(header.rows, header.cols, header.type);

        size_t bytes = mat.rows * mat.cols * mat.elemSize();
        ret = fread(static_cast<void*>(mat.data), 1, bytes, file.get());
        assert(ret > 0);
        cache.insert(index, mat);
        return mat;*/
        assert(false);
        return cv::Mat();
    } else {
        MatHeader header = *reinterpret_cast<MatHeader*>(&data[offset]);
        cv::Mat mat(header.rows, header.cols, header.type);
        mat.data = reinterpret_cast<unsigned char*>(&data[offset + sizeof(MatHeader)]);
        return mat;
    }
}

bool VecMatPersistor::contains(const uint64_t index) {
    return lookup.find(index) != lookup.end();
}

std::vector<VecMatPersistor::DescriptorInfo> VecMatPersistor::getDescriptorInfo() const {
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