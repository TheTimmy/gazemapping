#pragma once
#include <vector>
#include <string>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "Catalog.h"

// void compute_features(const std::vector<std::string>& video_filenames, const std::string& databaseFile);
void compute_descriptors(const Catalog& catalog, boost::property_tree::ptree const& config);