#include <vector>
#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>


#include "Catalog.h"
#include "Features.h"
#include "Fixation.h"
#include "VocabularyTree.h"

#define _USE_MATH_DEFINES
#include <math.h>

namespace po = boost::program_options;

template <typename T>
std::vector<T> as_vector(boost::property_tree::ptree const& pt, boost::property_tree::ptree::key_type const& key) {
    std::vector<T> r;
    for (auto& item : pt.get_child(key))
        r.push_back(item.second.get_value<T>());
    return r;
}

boost::property_tree::ptree load_config(const std::string& filename) {
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(filename, pt);
    return pt;
}

Catalog load_catalog(boost::property_tree::ptree const& pt) {
    auto timestamps = as_vector<size_t>(pt, "timestamps");
    auto gaze       = as_vector<std::string>(pt, "gaze");
    auto videos     = as_vector<std::string>(pt, "videos");
    auto vocabPath = pt.get<std::string>("vocabulary");
    auto tracker_rate = pt.get<double>("tracker_rate");
    return Catalog(videos, gaze, timestamps, vocabPath, tracker_rate);
}

void compute_vocabulary_tree(Catalog& catalog, boost::property_tree::ptree const &config, const std::string& filename) {
    const int subsample = config.get<int>("subsample");
    const int clusters = config.get<int>("clusters");
    const int height = config.get<int>("height");
    const int batchSize = config.get<int>("batch_size");
    const int minDescriptors = config.get<int>("min_descriptors");
    const int max_iterations = config.get<int>("max_iterations");
    const int max_no_improvements = config.get<int>("max_no_improvements");
    const double tolerance = config.get<double>("tolerance");
    const bool in_memory = config.get<bool>("in_memory");

    VocabularyTree tree(clusters, height, batchSize, minDescriptors, max_iterations, max_no_improvements, tolerance, catalog, filename);
    tree.create(subsample, in_memory);
    catalog.unload();
}

void compute_fixation(Catalog& catalog, boost::property_tree::ptree const& config, const std::string& filename, EvaluationMethod eval) {
    // auto angle2Screen = +[](const GazePoint& p) {
    //     return p;
    //     constexpr double verticalFov = M_PI * 110.0 / 180.0;
    //     constexpr double screenWidth = 1080.0;
    //     constexpr double screenHeight = 1200.0;
    //     const double screenDist = 0.5 * screenHeight / std::tanh(verticalFov / 2.0);
    //     const double x = (screenDist * std::tan( M_PI * p.x / 180.0) + 0.5 * screenWidth) / screenWidth;
    //     const double y = (screenDist * std::tan(-M_PI * p.y / 180.0) + 0.5 * screenHeight) / screenHeight;
    //     return GazePoint { static_cast<float>(x), static_cast<float>(y) };
    // };

    const bool in_memory = config.get<bool>("in_memory");
    catalog.prefetch(true, true, in_memory, nullptr);
    VocabularyTree tree(catalog, filename);
    Fixation fix(catalog, tree);

    auto directory = boost::filesystem::path(config.get<std::string>("output_directory"));
    if (!boost::filesystem::exists(directory)) {
        boost::filesystem::create_directory(directory);
    }

    auto indices = catalog.getVideoIndices();
    for (auto index : indices) {
        auto filename = directory / boost::filesystem::path("Saliency-" + std::to_string(index) + ".avi");
        if (boost::filesystem::exists(filename)) {
            std::cout << "Skip Fixation: " << index << std::endl;
        } else {
            std::cout << "Compute Fixation: " << index << std::endl;
            fix.computeFixation(filename.string(), static_cast<uint32_t>(index), config, eval);
        }
    }
}

int main(int argc, char* argv[]) {
    po::options_description config("Global options");
    config.add_options()
        ("command", po::value<std::string>()->required(), "Command to execute (desc, tree, fix)")
        ("catalog", po::value<std::string>()->required(), "The video, gaze, scene label and timestamp data to create the descriptors from.")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, config), vm);
    po::notify(vm);

    //po::store(parsed, vm);
    std::string cmd = vm["command"].as<std::string>();
    if (cmd == "desc") {
        std::cout << "Compute Descriptors" << std::endl;
        auto config = load_config(vm["catalog"].as<std::string>());

        Catalog catalog = load_catalog(config);
        auto pt = config.get_child("features");
        compute_descriptors(catalog, pt);
    }
    else if (cmd == "tree") {
        std::cout << "Compute Vocabulary Tree" << std::endl;
        auto config = load_config(vm["catalog"].as<std::string>());

        Catalog catalog = load_catalog(config);
        auto pt = config.get_child("tree");
        compute_vocabulary_tree(catalog, pt, config.get<std::string>("filename"));
    }
    else if (cmd == "fixation") {
        std::cout << "Compute Fixations" << std::endl;
        auto config = load_config(vm["catalog"].as<std::string>());
        Catalog catalog = load_catalog(config);
        auto pt = config.get_child("fixations");
        auto validation = config.get<std::string>("evaluation");

        EvaluationMethod eval = EvaluationMethod::None;
        if (validation == "cross validation") {
            eval = EvaluationMethod::CrossValidation;
        } else if (validation == "temporal cross validation") {
            eval = EvaluationMethod::TemporalCrossValidation;
        } else if (validation == "past temporal cross validation") {
            eval = EvaluationMethod::PastTemporalCrossValidation;
        }
        
        std::cout << "Evaluate using: " << eval << std::endl;
        compute_fixation(catalog, pt, config.get<std::string>("filename"), eval);
    }
    else {
        std::cout << "Unknown command: " << cmd << std::endl;
    }

    return 0;
}
