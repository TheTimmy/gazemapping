#pragma once
#include <opencv2/core.hpp>
#include <memory>
#include <vector>
#include <deque>
#include <atomic>
#include <thread>

#include "Pool.h"

class MiniBatchKMeans {
public:
    MiniBatchKMeans(int clusters, int iterations=100, int batchSize=10000, size_t max_no_improvement=10, double tolerance=1e-7);

    void fit(cv::Mat& desc);
    void label(const cv::Mat& desc, cv::Mat& labels);
    void label(const cv::Mat& desc, std::vector<uint8_t>& labels);
    void distance(const cv::Mat& desc, std::vector<double>& labels);
    bool converged();

    double error() const;
    double error_diff() const;

    cv::Mat centers;
private:
    void initClusters(cv::Mat& desc);

    std::vector<size_t> N;
    
    std::atomic<double> interna;
    double min_interna, diff_interna, last_interna = -1.0;
    const double tolerance;

    size_t iteration = 0;
    size_t approxSampleCount = 0;
    size_t interna_improvement = 0;
    const size_t max_no_improvement;

    const int clusters;
    const int iterations;
    const int batchSize;
    bool initialized;
};