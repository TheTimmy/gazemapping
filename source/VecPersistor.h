#pragma once

#include <stdlib.h>
#include <iostream>
#include <vector>
#include <fstream>

#include <opencv2/core.hpp>


class VecPersistor {
public:
    /**
     * Persists the given a vector vec to a file filePath
     * @tparam T data type
     * @param filePath output file path
     * @param vec input vector
     */
    template<typename T>
    static void persist(const std::string& filePath, std::vector<T>& vec);

    template<typename T>
    static void persist(const std::string& filePath, std::vector<std::vector<T>>& vec);

    template<typename T>
    static void persist(const std::string& filePath, std::vector<std::vector<std::vector<T>>>& vec);

    static void persist(const std::string& filePath, std::vector<std::vector<cv::KeyPoint>>& vec);

    /**
     * Given a file path, restores the file to a vector
     * @tparam T data type
     * @param filePath input file path
     * @param vec output vector
     */
    template<typename T>
    static void restore(const std::string& filePath, std::vector<T>& vec);

    template<typename T>
    static void restore(const std::string& filePath, std::vector<std::vector<T>> &vec);
};

template<typename T>
void VecPersistor::persist(const std::string& filePath, std::vector<T>& vec) {
    static_assert(std::is_pod_v<T>, "T must be pod");

    std::ofstream file(filePath.c_str(), std::ios::out | std::ios::binary);

    size_t size = vec.size();
    file.write(reinterpret_cast<char*>(&size), sizeof(size_t));
    file.write(reinterpret_cast<char*>(&vec[0]), size * sizeof(T));
}

template<typename T>
void VecPersistor::persist(const std::string& filePath, std::vector<std::vector<std::vector<T>>>& vec) {
    static_assert(std::is_pod_v<T>, "T must be pod");

    std::ofstream file(filePath.c_str(), std::ios::out | std::ios::binary);

    size_t size = vec.size();
    file.write(reinterpret_cast<char*>(&size), sizeof(size_t));
    for(size_t i = 0; i < size; ++i) {
        size_t middlesize = vec.size();
        file.write(reinterpret_cast<char*>(&middlesize), sizeof(size_t));

        for (size_t j = 0; j < middlesize; ++j) {
            size_t innerSize = vec[i].size();
            file.write(reinterpret_cast<char*>(&innerSize), sizeof(size_t));
            file.write(reinterpret_cast<char*>(&vec[i][j][0]), innerSize * sizeof(T));
        }
    }
}


template<typename T>
void VecPersistor::persist(const std::string& filePath, std::vector<std::vector<T>>& vec) {
    static_assert(std::is_pod_v<T>, "T must be pod");

    std::ofstream file(filePath.c_str(), std::ios::out | std::ios::binary);

    size_t size = vec.size();
    file.write(reinterpret_cast<char*>(&size), sizeof(size_t));
    for(size_t i = 0; i < size; ++i) {
        size_t innerSize = vec[i].size();
        file.write(reinterpret_cast<char*>(&innerSize), sizeof(size_t));
        file.write(reinterpret_cast<char*>(&vec[i][0]), innerSize * sizeof(T));
    }
}

template<typename T>
void VecPersistor::restore(const std::string& filePath, std::vector<T>& vec) {
    static_assert(std::is_pod_v<T>, "T must be pod");

    std::ifstream file(filePath.c_str(), std::ios::in | std::ios::binary);

    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));

    vec.resize(size);
    file.read(reinterpret_cast<char*>(&vec[0]), size * sizeof(T));
}

template<typename T>
void VecPersistor::restore(const std::string& filePath, std::vector<std::vector<T>>& vec) {
    static_assert(std::is_pod_v<T>, "T must be pod");

    std::ifstream file(filePath.c_str(), std::ios::in | std::ios::binary);

    size_t size = 0;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    vec.resize(size);
    for (size_t i = 0; i < size; ++i) {
        size_t innerSize;
        file.read(reinterpret_cast<char*>(&innerSize), sizeof(innerSize));
        vec[i].resize(innerSize);
        file.read(reinterpret_cast<char*>(&vec[i][0]), innerSize * sizeof(T));
    }
}
