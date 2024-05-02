#pragma once

#include "Catalog.h"
#include "VocabularyTree.h"

#include "Pool.h"
#include <boost/property_tree/ptree.hpp>

enum EvaluationMethod {
    None = 0,
    CrossValidation = 1,
    TemporalCrossValidation = 2,
    PastTemporalCrossValidation = 3
};

cv::Mat filterKeyPoints(const cv::Size2i& size, std::vector<cv::KeyPoint>& kp, const cv::Mat descriptors,
                        const std::vector<GazePoint>& gaze, const cv::Size2i& patchSize,
                        const int MIN_MATCH_COUNT, const bool assignNewKeypoints=true);

void filterMatches(const std::vector<std::vector<cv::DMatch>>& original, std::vector<cv::DMatch>& filtered, float ratio);

class Fixation {
public:
    Fixation(Catalog& catalog, VocabularyTree& tree);

    void computeFixation(const std::string& path, uint32_t videoIndex, boost::property_tree::ptree const& config, EvaluationMethod cross_validate);

private:
    Catalog& catalog;
    VocabularyTree& tree;
};