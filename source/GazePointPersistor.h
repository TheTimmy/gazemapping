#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <math.h>
#include <functional>

struct GazePoint {
    GazePoint() = default;
    GazePoint(const GazePoint&) = default;
    GazePoint& operator = (const GazePoint&) = default;
    ~GazePoint() = default;

    float x;
    float y;
};

class GazePointPersistor {
public:
    GazePointPersistor() = default;
    GazePointPersistor(GazePointPersistor&&);
    GazePointPersistor(const GazePointPersistor& o) = delete;
    GazePointPersistor(const std::string& filename, uint64_t timestamp, double framerate, std::function<GazePoint (const GazePoint&)> converter = nullptr);
    ~GazePointPersistor() = default;

    GazePointPersistor& operator = (GazePointPersistor&& o);
    GazePointPersistor& operator = (const GazePointPersistor& o) = delete;

    void close();
    bool isOpen() const { return is_open; }
    void read(const std::string& filename, uint64_t timestamp);

    static GazePointPersistor open(const std::string& filename, size_t timestamp, double framerate, std::function<GazePoint (const GazePoint&)> converter);

    const std::vector<GazePoint> operator [] (size_t index) const { 
        std::vector<GazePoint> gaze;
        size_t start = static_cast<size_t>(static_cast<double>(index) * framerate);
        size_t stop  = static_cast<size_t>(std::round(static_cast<double>(index + 1) * framerate));

        start = std::min(std::max(start, 0ULL), gazes.size() - 1);
        stop  = std::min(std::max( stop, 0ULL), gazes.size() - 1);
        for (size_t i = start; i <= std::min(gazes.size() - 1, stop); ++i) {
            gaze.push_back(gazes[i]);
        }
        return gaze;
    }

private:
    std::vector<GazePoint> gazes;
    std::function<GazePoint (const GazePoint&)> converter;
    double framerate;
    bool is_open = false;
};
