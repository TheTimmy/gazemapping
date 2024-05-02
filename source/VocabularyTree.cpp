#include <iostream>
#include <numeric>

#include "RobinHood.h"
#include "VocabularyTree.h"
#include "MiniBatchKMeans.h"
#include "VecPersistor.h"
#include "MatPersistor.h"
#include "KMeans.h"

VocabularyTree::VideoDescriptor VocabularyTree::VideoDescriptor::INVALID = { std::numeric_limits<decltype(frame)>::max(), std::numeric_limits<decltype(video)>::max() };

size_t computeTotalSize(int clusters, int depth, size_t featureCount) {
    size_t total = featureCount;
    if (depth == 0)
        return total;

    for (int i = 0; i < clusters; ++i) {
        total += computeTotalSize(clusters, depth - 1, featureCount / clusters);
    }
    return total;
}

VocabularyTree::VocabularyTree(Catalog& catalog, const std::string& prefix) 
    : catalog(catalog), prefix(prefix) {
    VecPersistor::restore(prefix + ".index", index);
    VecPersistor::restore(prefix + ".leaves", leaves);
    VecPersistor::restore(prefix + ".comps", components);

    MatPersistor pers1(prefix + ".centers"); pers1.openRead(); pers1.read(centers);
    MatPersistor pers2(prefix + ".weights"); pers2.openRead(); pers2.read(weights);

    // replace me
    FileStorage file(prefix + ".info", cv::FileStorage::READ);
    clusters         = (int) file["clusters"];
    height           = (int) file["depth"];
    dbSize           = (int) file["dbSize"];
    nodeCount        = (int) file["nNodes"];
    usedNodes        = (int) file["nextIdNode"];
    usedLeaves       = (int) file["nextIdLeaf"];
    totalDescriptors = (int) file["totDescriptors"];

    imageCount = 0;
    const auto indices = catalog.getVideoIndices();
    videoCount = indices.size();
    for (auto index : indices) {
        auto info = catalog.getVideoInfo(index);
        imageCount = std::max((size_t)info[info.size() - 1].frame + 1, imageCount);
    }
}

VocabularyTree::VocabularyTree(int clusters, int height, int batchSize, int minDescriptors, int max_iterations, int max_no_improvements, double tolerance, Catalog& catalog, const std::string& prefix) 
    : clusters(clusters), height(height), batchSize(batchSize), catalog(catalog), minDescriptors(minDescriptors), prefix(prefix), max_iterations(max_iterations), max_no_improvements(max_no_improvements), tolerance(tolerance) {
}

void VocabularyTree::query(cv::Mat image, std::vector<Matching>& matching, double threshold) {
    // std::unordered_map<uint32_t, float> q;
    std::vector<float> q(nodeCount, 0.0f);

    double sum = 0.0;
    std::vector<uint32_t> path;
    for (size_t i = 0; i < image.rows; i++) {
        cv::Mat descriptor = image.row(i);
        findPath(descriptor, path);

        for (uint32_t nodeID : path) {
            const uint32_t nodeIndex = index[nodeID];
            const float weight = weights.at<float>(nodeIndex);
            if (weight > 0.0f && !std::isinf(weight)) {
                q[nodeIndex] += weight;
                sum += weight;
            }
        }
    }

    for (auto& qi : q) {
        qi /= sum;
    }

    //std::unordered_map<VideoDescriptor, Matching, VideoHash> matches; 
    //robin_hood::unordered_map<VideoDescriptor, Matching, VideoHash> matches;
    matching.clear();
    matching.resize(videoCount * imageCount, Matching());

    for (size_t i = 0; i < q.size(); ++i) {
        const float qi = q[i];
        if (qi > 0.0f) {
            for (const auto& c : components[i]) {
                const float di = c.value;
                const float diff = std::abs(qi - di);

                Matching& match = matching[c.descriptor.video * imageCount + c.descriptor.frame];
                match.frame = c.descriptor.frame;
                match.video = c.descriptor.video;
                match.score += (diff - di - qi);
                match.count++;
            }
        }
    }

    /*for (const auto& m : matches) {
        matching.push_back(m.second);
    }*/
    matching.erase(std::remove_if(std::begin(matching), std::end(matching), [threshold] (const auto& match) {
        return match.count == 0 || match.score > -threshold; 
    }), matching.end());

    std::sort(matching.begin(), matching.end(), [](const auto& a, const auto& b) {
        return a.score < b.score;
    });
}

void VocabularyTree::create(int subsample_frames, bool in_memory) {
    std::vector<VideoIndex> fileDescriptors = catalog.constructDescriptors(subsample_frames, in_memory);
    const auto size = fileDescriptors.size();
    std::cout << "Construct Tree for " << size << " descriptors" << std::endl;

    std::cout << "Allocate Clusters" << std::endl;
    allocateNodes();

    std::cout << "Compute Clusters" << std::endl;
    createNode(0, 0, fileDescriptors);

    std::cout << "\nSave Clusters" << std::endl;
    storeNodes();

    std::cout << "Compute Inverted Index" << std::endl;
    computeInvertedIndex();

    std::cout << "\nStore Inverted Index" << std::endl;
    storeInvertedIndex();

    computeVectors();
    std::cout << "\n\n\n" << std::endl;

    std::cout << "-----------------------------" << std::endl;
    std::cout << "VocTree Info: " << std::endl;
    std::cout << ">max height (H): " << height << std::endl;
    std::cout << ">children by node (K): " << clusters << std::endl;
    std::cout << ">DB file count: " << dbSize << std::endl;
    std::cout << ">total nodes: " << usedNodes << std::endl;
    std::cout << ">total leaves: " << usedLeaves << std::endl;
    std::cout << "-----------------------------" << std::endl;
}

void VocabularyTree::createNode(int nodeID, int level, std::vector<VideoIndex>& fileDescriptors) {
    uint32_t nodeIndex = getNextIdxNode();
    index[nodeID] = nodeIndex;

    if (level == height || fileDescriptors.size() <= minDescriptors) {
        leaves[nodeIndex] = getNextIdxLeaf();
        return;
    }

    // cluster all features of the node
    size_t counter = 0;
    size_t iteration = 0;
    cv::Mat descriptors, lastDescriptor, clusterCenters;
    VideoIndex lastIndex = VideoIndex(-1, -1, -1);

    std::vector<uint8_t> labels;
    labels.reserve(fileDescriptors.size());

    if (fileDescriptors.size() >= batchSize) {
        MiniBatchKMeans kmeans(clusters, max_iterations, batchSize, max_no_improvements, tolerance);
        while (!kmeans.converged()) {
            counter = 0;
            iteration++;
            for (const VideoIndex& index : fileDescriptors) {
                if (counter++ % (1 << 18) == 0) {
                    std::cout << '\r' << "Cluster: " << counter << "/" << fileDescriptors.size() << " @ " << iteration << "   " << kmeans.error_diff() << "             " << std::flush; //"%              " << std::flush;
                }

                cv::Mat desc = catalog.getDescriptor(index);
                if (descriptors.empty()) {
                    descriptors = desc;
                }
                else {
                    descriptors.push_back(desc);
                }

                if (descriptors.rows >= batchSize) {
                    cv::Mat fDescriptors;
                    descriptors.convertTo(fDescriptors, CV_32F);

                    kmeans.fit(fDescriptors);
                    descriptors = cv::Mat();
                }
            }

            if (descriptors.rows != 0) {
                cv::Mat fDescriptors;
                descriptors.convertTo(fDescriptors, CV_32F);
                kmeans.fit(fDescriptors);
                descriptors = cv::Mat();
            }
        }

        // label all features
        counter = 0;
        descriptors = cv::Mat();
        for (const VideoIndex& index : fileDescriptors) {
            if (counter++ % (1 << 18) == 0) {
                std::cout << '\r' << "Cluster: " << counter << "/" << fileDescriptors.size() << " @ " << iteration << "   " << kmeans.error_diff() << "             " << std::flush; //"%              " << std::flush;
            }

            cv::Mat desc = catalog.getDescriptor(index);
            if (descriptors.empty()) {
                descriptors = desc;
            }
            else {
                descriptors.push_back(desc);
            }

            if (descriptors.rows >= batchSize) {
                cv::Mat fDescriptors;
                descriptors.convertTo(fDescriptors, CV_32F);
                kmeans.label(fDescriptors, labels);
                descriptors = cv::Mat();
            }
        } 

        if (descriptors.rows != 0) {
            cv::Mat fDescriptors;
            descriptors.convertTo(fDescriptors, CV_32F);
            kmeans.label(fDescriptors, labels);
        }

        clusterCenters = kmeans.centers;
    } else {
        counter = 0;
        for (const VideoIndex& index : fileDescriptors) {
            if (counter++ % (1 << 14) == 0) {
                std::cout << '\r' << "Cluster: " << counter << "/" << fileDescriptors.size() << "             " << std::flush; //"%              " << std::flush;
            }

            cv::Mat desc = catalog.getDescriptor(index);
            if (descriptors.empty()) {
                descriptors = desc;
            } else {
                descriptors.push_back(desc);
            }
        }

        std::cout << '\r' << "Cluster: " << fileDescriptors.size() << "/" << fileDescriptors.size() << "    " << static_cast<double>(usedNodes) / nodeCount << "%              " << std::flush;

        cv::Mat fDescriptors;
        descriptors.convertTo(fDescriptors, CV_32F);

        KMeans kmeans(clusters, max_iterations, batchSize);
        kmeans.fit(fDescriptors);
        kmeans.label(fDescriptors, labels);
        clusterCenters = kmeans.centers;
    }

    // prepare new accessor indices for the child nodes
    std::vector<std::vector<VideoIndex>> newDescriptors;
    newDescriptors.resize(clusters);

    for (size_t i = 0; i < fileDescriptors.size(); ++i) {
        newDescriptors[labels[i]].push_back(fileDescriptors[i]);
    }

    // clear memory to avoid keeping things multiple times
    labels.clear();
    labels.shrink_to_fit();
    fileDescriptors.clear();
    fileDescriptors.shrink_to_fit();

    // process childs
    leaves[nodeIndex] = std::numeric_limits<decltype(leaves)::value_type>::max();
    for (int i = 0; i < clusters; ++i) {
        int newLevel = level + 1;
        uint32_t childID = computeChildID(nodeID, i);

        createNode(childID, newLevel, newDescriptors[i]);

        uint32_t childIndex = index[childID];
        clusterCenters.row(i).copyTo(centers.row(childIndex));
    }
}

void VocabularyTree::allocateNodes() {
    usedNodes = 0;
    usedLeaves = 0;
    totalDescriptors = 0;

    nodeCount = (pow(clusters, height + 1) - 1) / (clusters - 1);

    // fills the indexes with 0s
    index.assign(nodeCount, 0);
    leaves.resize(nodeCount);

    // create center buffer and expands it
    // for example. (K=10, h=6, dim=128x4) => approx. 543 MB
    centers.create(nodeCount + 1, 128, CV_32F);
}

void VocabularyTree::storeNodes() {
    leaves.resize(usedNodes);

    VecPersistor::persist(prefix + ".index", index);
    VecPersistor::persist(prefix + ".leaves", leaves);

    MatPersistor pers(prefix + ".centers");
    centers.resize(usedNodes);
    pers.create(centers);
}

bool VocabularyTree::isLeaf(uint32_t id) {
    uint32_t nodeID = index[id];
    return leaves[nodeID] != std::numeric_limits<decltype(leaves)::value_type>::max();
}

uint32_t VocabularyTree::findLeaf(cv::Mat descriptor) {
    uint32_t nodeID = 0;
    while(!isLeaf(nodeID)) {
        int minChildID = computeChildID(nodeID, 0);
        double minDistance = cv::norm(descriptor, centers.row(index[minChildID]), 4);

        for (size_t i = 1; i < clusters; ++i) {
            uint32_t childID = computeChildID(nodeID, i);
            uint32_t childIndex = index[childID];

            const double distance = cv::norm(descriptor, centers.row(childIndex), 4);
            if (distance < minDistance) {
                minDistance = distance;
                minChildID = childID;
            }
        }

        nodeID = minChildID;
    }

    return nodeID;
}

void VocabularyTree::findPath(cv::Mat descriptor, std::vector<uint32_t>& path) {
    path.clear();

    uint32_t nodeID = 0;
    path.push_back(nodeID);
    while(!isLeaf(nodeID)) {
        int minChildID = computeChildID(nodeID, 0);
        int minChildIndex = index[minChildID];
        double minDistance = cv::norm(descriptor, centers.row(minChildIndex), 4);

        for (size_t i = 1; i < clusters; ++i) {
            uint32_t childID = computeChildID(nodeID, i);
            uint32_t childIndex = index[childID];

            const double distance = cv::norm(descriptor, centers.row(childIndex), 4);
            if (distance < minDistance) {
                minDistance = distance;
                minChildID = childID;
            }
        }

        nodeID = minChildID;
        path.push_back(nodeID);
    }
}

void VocabularyTree::computeInvertedIndex() {
    invertedIndex.resize(usedLeaves);

    std::cout << "Load Inverted Index Descriptors" << std::endl;
    auto imageDescriptors = catalog.constructImageDescriptors();

    dbSize = imageDescriptors.size();
    for (size_t i = 0; i < dbSize; ++i) {
        std::cout << "\rInvert Index: " << i << "/" << dbSize << "        ";

        cv::Mat fDescriptor;
        cv::Mat descriptor = catalog.getImageDescriptors(imageDescriptors[i]);
        descriptor.convertTo(fDescriptor, CV_32F);

        for (size_t j = 0; j < fDescriptor.rows; ++j) {
            const uint32_t leafID = findLeaf(fDescriptor.row(j));
            const uint32_t nodeIndex = index[leafID];
            const uint32_t leafIndex = leaves[nodeIndex];

            invertedIndex[leafIndex].push_back(VideoDescriptor {
                static_cast<uint16_t>(imageDescriptors[i].frame),
                imageDescriptors[i].video
            });

            totalDescriptors++;
        }
    }
}

void VocabularyTree::storeInvertedIndex() {
    VecPersistor::persist(prefix + ".inv", invertedIndex);
}

void VocabularyTree::computeVectors() {
    weights.create(usedNodes, 1, CV_32F);
    components.resize(usedNodes);

    std::unordered_set<Entry, EntryHash> entries;
    computeInvertedIndex(0, 0, entries);

    std::unordered_map<VideoDescriptor, double, VideoHash> sums;
    for (size_t i = 0; i < components.size(); ++i) {
        const auto& comps = components[i];
        for (size_t j = 0; j < comps.size(); ++j) {
            sums[comps[j].descriptor] += comps[j].value;
        }
    }

    for (size_t i = 0; i < components.size(); ++i) {
        auto& comps = components[i];
        for (size_t j = 0; j < comps.size(); ++j) {
            Component& dc = comps[j];
            dc.value /= sums[dc.descriptor];
        }
    }

    MatPersistor pers(prefix + ".weights");
    pers.create(weights);

    VecPersistor::persist(prefix + ".comps", components);

    // replace me
    FileStorage file(prefix + ".info", cv::FileStorage::WRITE);
    file << "clusters" << (int)clusters;
    file << "depth" << (int)height;
    file << "dbSize" << (int)dbSize;
    file << "nNodes" << (int)nodeCount;
    file << "nextIdNode" << (int)usedNodes;
    file << "nextIdLeaf" << (int)usedLeaves;
    file << "totDescriptors" << (int)totalDescriptors;
}

void VocabularyTree::computeInvertedIndex(uint32_t nodeID, uint32_t level, std::unordered_set<Entry, EntryHash>& entries) {
    if (isLeaf(nodeID)) {
        uint32_t nodeIndex = index[nodeID];
        uint32_t leafIndex = leaves[nodeIndex];

        Entry entry = { VideoDescriptor::INVALID, 0 };
        const auto& invIndex = invertedIndex[leafIndex];
        for (size_t i = 0; i < invIndex.size(); ++i) {
            const VideoDescriptor& elementID = invIndex[i];
            if (elementID != entry.descriptor) {
                if (entry.descriptor.valid()) {
                    entries.insert(entry);
                }
                entry.descriptor = elementID;
                entry.features = 1;
            } else {
                entry.features++;
            }
        }

        if (entry.descriptor.valid()) {
            entries.insert(entry);
        }
    } else {
        entries.clear();
        std::vector<std::unordered_set<Entry, EntryHash>> virtualEntries;
        virtualEntries.resize(clusters);

        for (size_t i = 0; i < clusters; ++i) {
            uint32_t childID = computeChildID(nodeID, i);
            computeInvertedIndex(childID, level + 1, virtualEntries[i]);
        }

        for (size_t i = 0; i < clusters; ++i) {
            for (const auto& entry : virtualEntries[i]) {
                auto found = entries.find(entry);
                if (found != entries.end()) {
                    Entry newEntry = *found;
                    newEntry.features += entry.features;
                    entries.insert(newEntry);
                } else {
                    entries.insert(entry);
                }
            }
            virtualEntries[i].clear();
        }
    }

    size_t Ni = entries.size();
    size_t N  = dbSize;

    const double weight = std::log(static_cast<double>(N) / static_cast<double>(Ni));
    const uint32_t nodeIndex = index[nodeID];
    weights.at<float>(nodeIndex) = weight;

    auto& comps = components[nodeIndex];
    comps.resize(Ni);

    size_t counter = 0;
    for (const auto& entry : entries) {
        double featCount = static_cast<double>(entry.features);
        comps[counter++] = Component { entry.descriptor, static_cast<float>(weight * featCount) };
    }
}


uint32_t VocabularyTree::getNextIdxNode() {
    uint32_t ret = usedNodes;
    usedNodes++;
    return ret;
}

uint32_t VocabularyTree::getNextIdxLeaf() {
    uint32_t ret = usedLeaves;
    usedLeaves++;
    return ret;
}

uint32_t VocabularyTree::computeChildID(uint32_t nodeID, uint32_t childNum) {
    return (static_cast<uint32_t>(clusters) * nodeID) + childNum + 1;
}
