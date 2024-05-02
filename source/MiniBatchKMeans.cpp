#include "MiniBatchKMeans.h"

#include <limits>
#include <random>
#include <opencv2/core.hpp>

#define CV_L2 4

MiniBatchKMeans::MiniBatchKMeans(int clusters, int iterations, int batchSize, size_t max_no_improvement, double tolerance) 
    : max_no_improvement(max_no_improvement), tolerance(tolerance), clusters(clusters), iterations(iterations), batchSize(batchSize), initialized(false) {
}

bool MiniBatchKMeans::converged() {
    if (!initialized)
        return false; // we havent initialized yet so lets see first

    if (++iteration >= iterations)
        return true;

    double cinterna = interna.load();
    if (cinterna < min_interna) {
        min_interna = cinterna;
        interna_improvement = 0;
    } else {
        if (++interna_improvement == max_no_improvement) {
            return true;
        }
    }

    if (last_interna < 0.0) {
        diff_interna = std::numeric_limits<double>::max();
        last_interna = cinterna;
    } else {
        diff_interna = std::abs(cinterna - last_interna);
        last_interna = cinterna;
    }

    if (diff_interna < tolerance) {
        return true;
    }

    interna.store(0);

    return false;
}

double MiniBatchKMeans::error() const {
    return interna;
}

double MiniBatchKMeans::error_diff() const {
    return diff_interna;
}

void MiniBatchKMeans::initClusters(cv::Mat& desc) {
    thread_pool& pool = thread_pool::instance();
    auto distance = [this, &pool](const size_t numInitClusters, const cv::Mat& desc, std::vector<double>& distances) {
        const size_t count = std::ceil(static_cast<double>(desc.rows) / pool.threadCount());

        distances.clear();
        distances.resize(desc.rows);

        std::vector<std::future<double>> done;
        done.resize(pool.threadCount());

        for (size_t t = 0; t < pool.threadCount(); ++t) {
            size_t start = t * count;
            size_t end   = std::min((t + 1) * count, (size_t)desc.rows);

            done[t] = pool.enqueue_task(
                [this, numInitClusters, start, end, &distances, &desc]() {
                    double sum = 0.0;
                    for (size_t i = start; i < end; ++i) {
                        cv::Mat row = desc.row(i);

                        double distance = std::numeric_limits<double>::max();
                        for (size_t k = 0; k < numInitClusters; ++k) {
                            cv::Mat cluster = centers.row(k);
                            cv::Mat diff = cluster - row;

                            const double norm = cv::norm(diff, CV_L2);
                            distance = std::min(distance, norm);
                        }

                        distances[i] = distance;
                        sum += distance;
                    }
                    return sum;
                }
            );
        }

        double sum = 0.0;
        for (int t = 0; t < pool.threadCount(); t++) {
            done[t].wait();
            sum += done[t].get();
        }

        return sum;
    };

    N.resize(clusters);
    for (int i = 0; i < clusters; ++i)
        N[i] = 0;
        
    centers = cv::Mat(clusters, desc.cols, CV_32F);
    centers = 0;

    std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> distribution(0, desc.rows - 1);
    std::uniform_real_distribution<float> fdistribution(0.0, 1.0);
    desc.row(distribution(gen)).copyTo(centers.row(0));

    size_t numInitClusters = 1;
    std::vector<double> distances;
    for (int j = 1; j < clusters; ++j) {
        size_t index = 0;
        const double probability = fdistribution(gen);
        double sum = distance(numInitClusters, desc, distances) * probability;
        while (sum > 0.0) {
            sum -= distances[index++];
        }

        desc.row(std::min(index, static_cast<size_t>(desc.rows) - 1)).copyTo(centers.row(j));
        numInitClusters++;
    }
}

void MiniBatchKMeans::fit(cv::Mat& desc) {
    if (!initialized) {
        initialized = true;
        interna_improvement = 0;
        min_interna = std::numeric_limits<double>::max();
        initClusters(desc);
    }

    thread_pool& pool = thread_pool::instance();
    std::vector<std::future<bool>> done;
    done.resize(pool.threadCount());

    std::vector<uint8_t> labels;
    labels.resize(std::min(desc.rows, batchSize));
    if (iteration == 0) {
        approxSampleCount += desc.rows;
    }

    // compute distances and cluster indices
    for (int t = 0; t < pool.threadCount(); ++t) {
        const size_t count = std::ceil(static_cast<double>(labels.size()) / pool.threadCount());
        const size_t start = t * count;
        const size_t end   = std::min((t + 1) * count, (size_t)labels.size());

        done[t] = pool.enqueue_task(
            [this, start, end, &labels, &desc]() {
                for (size_t j = start; j < end; ++j) {
                    const cv::Mat row = desc.row(j);

                    size_t index = 0;
                    double distance = std::numeric_limits<double>::max();
                    for (size_t k = 0; k < clusters; ++k) {
                        const cv::Mat cluster = centers.row(k);
                        const double norm = cv::norm(cluster - row, CV_L2);
                        if (norm < distance) {
                            index = k;
                            distance = norm;
                        }
                    }

                    labels[j] = index;

                    double value;
                    do {
                        value = interna.load();
                    } while (!interna.compare_exchange_strong(value, value + distance,
                                                              std::memory_order_release,
                                                              std::memory_order_relaxed));
                }

                return true;
            }
        );
    }

    for (int t = 0; t < pool.threadCount(); ++t) {
        done[t].wait();
    }

    for(size_t i = 0; i < labels.size(); ++i) {
        const auto clusterIndex = labels[i];
        cv::Mat center = centers.row(clusterIndex);

        N[clusterIndex]++;
        const double eta = 1.0 / static_cast<double>(N[clusterIndex]);
        center = (1.0 - eta) * center + eta * desc.row(i);
    }

    // update cluster centers
    /*for (int t = 0; t < pool.threadCount(); ++t) {
        const size_t count = std::ceil(static_cast<double>(labels.size()) / pool.threadCount());
        const size_t start = t * count;
        const size_t end   = std::min((t + 1) * count, (size_t)labels.size());

        done[t] = pool.enqueue_task(
            [this, start, end, &labels, &desc]() {
                for (size_t j = start; j < end; ++j) {
                    const int clusterIndex = labels[j];
                    cv::Mat center = centers.row(clusterIndex);

                    std::unique_lock<std::mutex> lock(mutexes[clusterIndex]);
                    N[clusterIndex]++;
                    const double eta = 1.0 / static_cast<double>(N[clusterIndex]);
                    center = (1.0 - eta) * center + eta * desc.row(j);
                }
                return true;
            }
        );
    }

    for (int t = 0; t < pool.threadCount(); ++t)
        done[t].wait();*/
}

void MiniBatchKMeans::label(const cv::Mat& desc, cv::Mat& labels) {
    labels.create(desc.rows, 1, CV_8U);

    for (size_t i = 0; i < desc.rows; ++i) {
        cv::Mat row = desc.row(i);

        size_t index = 0;
        double distance = std::numeric_limits<double>::max();
        for (size_t k = 0; k < clusters; ++k) {
            cv::Mat cluster = centers.row(k);
            cv::Mat diff = cluster - row;

            const double norm = cv::norm(diff, CV_L2);
            if (norm < distance) {
                index = k;
                distance = norm;
            }
        }

        labels.at<uint8_t>((int)i, 0) = (uint8_t)index;
    }
}

void MiniBatchKMeans::label(const cv::Mat& desc, std::vector<uint8_t>& labels) {
    thread_pool& pool = thread_pool::instance();

    const size_t initial_size = labels.size();
    const size_t count = std::ceil(static_cast<double>(desc.rows) / pool.threadCount());

    labels.resize(initial_size + desc.rows);
    std::vector<std::future<bool>> done;
    done.resize(pool.threadCount());

    for (size_t t = 0; t < pool.threadCount(); ++t) {
        size_t start = t * count;
        size_t end   = std::min((t + 1) * count, (size_t)desc.rows);

        done[t] = pool.enqueue_task(
            [this, initial_size, start, end, &labels, &desc]() {
                for (size_t i = start; i < end; ++i) {
                    cv::Mat row = desc.row(i);

                    size_t index = 0;
                    double distance = std::numeric_limits<double>::max();
                    for (size_t k = 0; k < clusters; ++k) {
                        cv::Mat cluster = centers.row(k);
                        cv::Mat diff = cluster - row;

                        const double norm = cv::norm(diff, CV_L2);
                        if (norm < distance) {
                            index = k;
                            distance = norm;
                        }
                    }

                    labels[initial_size + i] = (uint8_t)index;
                }
                return true;
            }
        );
    }

    for (int t = 0; t < pool.threadCount(); t++) {
        done[t].wait();
    }
}

void MiniBatchKMeans::distance(const cv::Mat& desc, std::vector<double>& distances) {
    thread_pool& pool = thread_pool::instance();
    const size_t initial_size = distances.size();
    const size_t count = std::ceil(static_cast<double>(desc.rows) / pool.threadCount());

    distances.resize(initial_size + desc.rows);
    std::vector<std::future<bool>> done;
    done.resize(pool.threadCount());

    for (size_t t = 0; t < pool.threadCount(); ++t) {
        size_t start = t * count;
        size_t end   = std::min((t + 1) * count, (size_t)desc.rows);

        done[t] = pool.enqueue_task(
            [this, initial_size, start, end, &distances, &desc]() {
                for (size_t i = start; i < end; ++i) {
                    cv::Mat row = desc.row(i);

                    double distance = std::numeric_limits<double>::max();
                    for (size_t k = 0; k < clusters; ++k) {
                        cv::Mat cluster = centers.row(k);
                        cv::Mat diff = cluster - row;

                        const double norm = cv::norm(diff, CV_L2);
                        distance = std::min(distance, norm);
                    }

                    distances[initial_size + i] = distance;
                }
                return true;
            }
        );
    }

    for (int t = 0; t < pool.threadCount(); t++) {
        done[t].wait();
    }
}
