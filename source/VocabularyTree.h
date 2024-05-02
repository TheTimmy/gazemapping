#pragma once

#include <limits>
#include <unordered_set>
#include <opencv2/core.hpp>

#include "Common.h"
#include "Catalog.h"
#include "Pool.h"

struct Matching {
    float score = 0.0f;
    uint32_t frame = -1;
    uint16_t video = -1;
    uint16_t count = 0;
};

struct CompactMatchingIndex {
    uint32_t frame = -1;
    uint16_t video = -1;
};

struct CompactMatchingScore {
    float score = 0.0f;
    uint16_t count = 0;
};

class VocabularyTree {
public:
    VocabularyTree(Catalog& catalog, const std::string& prefix);
    VocabularyTree(int clusters, int height, int batchSize, int minDescriptors, int max_iterations, int max_no_improvements, double tolerance, Catalog& catalog, const std::string& prefix);

    void create(int subsample_frames, bool in_memory);
    void query(cv::Mat image, std::vector<Matching>& matching, double threshold = -std::numeric_limits<double>::max());
    void query(cv::Mat image, std::vector<CompactMatchingIndex>& matchingIndices, std::vector<CompactMatchingScore>& matchingScores, double threshold = -std::numeric_limits<double>::max());

private:
    PACK(struct VideoDescriptor {
        bool valid() const { return frame != std::numeric_limits<decltype(frame)>::max() && video != std::numeric_limits<decltype(video)>::max(); }
        inline bool operator == (const VideoDescriptor& o) const { return frame == o.frame && video == o.video; }
        inline bool operator != (const VideoDescriptor& o) const { return frame != o.frame || video != o.video; }

        uint16_t frame;
        uint16_t video;
        static VideoDescriptor INVALID;
    });

    struct VideoHash {
        size_t operator() (const VideoDescriptor& video) const {
            const size_t mask = (1ULL << 32ULL) - 1;
            return std::hash<size_t>()(((static_cast<size_t>(std::hash<int32_t>()((int32_t)video.frame)) & mask) <<  0ULL) |
                                       ((static_cast<size_t>(std::hash<int32_t>()((int32_t)video.video)) & mask) << 32ULL));
        }
    };

    PACK(struct Component {
        VideoDescriptor descriptor;
        float value;
    });

    PACK(struct Entry {
        inline bool operator == (const Entry& o) const { return descriptor == o.descriptor; }
        inline bool operator != (const Entry& o) const { return descriptor != o.descriptor; }

        VideoDescriptor descriptor;
        uint32_t features;
    });

    struct EntryHash {
        size_t operator() (const Entry& entry) const {
            return VideoHash()(entry.descriptor);
        }
    };

    void allocateNodes();
    void createNode(int nodeID, int level, std::vector<VideoIndex>& accessors);
    void storeNodes();

    //void storeIndices(const std::vector<VideoIndex>& fileDescriptors);
    void computeInvertedIndex();
    void storeInvertedIndex();

    void computeVectors();
    void computeInvertedIndex(uint32_t nodeID, uint32_t level, std::unordered_set<Entry, EntryHash>& entries);

    uint32_t getNextIdxNode();
    uint32_t getNextIdxLeaf();
    uint32_t computeChildID(uint32_t nodeID, uint32_t childNum);

    bool isLeaf(uint32_t index);
    uint32_t findLeaf(cv::Mat descriptor);
    void findPath(cv::Mat descriptor, std::vector<uint32_t>& path);

    cv::Mat centers;
    cv::Mat weights;

    Catalog& catalog;
    std::string prefix;

    std::vector<std::vector<VideoDescriptor>> invertedIndex;
    std::vector<std::vector<Component>> components;

    std::vector<uint32_t> index;
    std::vector<uint32_t> leaves;

    uint64_t dbSize;
    uint64_t usedNodes;
    uint64_t usedLeaves;
    uint64_t totalDescriptors;
    uint64_t nodeCount;

    uint64_t imageCount;
    uint64_t videoCount;

    int clusters;
    int height;
    int batchSize;
    int minDescriptors;
    int max_iterations;
    int max_no_improvements;
    double tolerance;
};