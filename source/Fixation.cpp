#include "Fixation.h"
#include <algorithm>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "GazePointPersistor.h"

// #define PROBABILITY_MAP
// #define DEBUG_OUTPUT

struct VideoIndexHash {
    size_t operator() (const VideoIndex& video) const {
        const size_t mask = (1ULL << 32ULL) - 1;
        return ((static_cast<size_t>(std::hash<int32_t>()((int32_t)video.frame)) & mask) <<  0ULL) |
               ((static_cast<size_t>(std::hash<int32_t>()((int32_t)video.video)) & mask) << 32ULL);
    }
};

void filterMatches(const std::vector<std::vector<cv::DMatch>>& original, std::vector<cv::DMatch>& filtered, float ratio) {
    size_t i = 0;
    for (size_t i = 0; i < original.size(); ++i) {
        const auto suboriginal = original[i];
        for (size_t j = 1; j < suboriginal.size(); ++j) {
            if (suboriginal[0].distance < ratio * suboriginal[j].distance) {
                filtered.emplace_back(suboriginal[0]);
            }
        }
    }
}

bool alignable(const std::vector<cv::DMatch>& original, const int min_match_count) {
    return original.size() > min_match_count;
}

cv::Mat alignImage(cv::Mat image1, const std::vector<cv::KeyPoint>& trainKeypoints, const std::vector<GazePoint>& trainGaze,
                   cv::Mat image2_2, cv::Mat image2, const std::vector<cv::KeyPoint>& queryKeypoints, const std::vector<GazePoint>& queryGaze,
                   const std::vector<cv::DMatch>& matches,
                   std::vector<GazePoint>& projectedGaze,
                   const float projection_threshold,
                   const int min_match_count) {
    if (!alignable(matches, min_match_count))
        return cv::Mat();

    std::vector<cv::Point2f> trainPoints, queryPoints;
    trainPoints.resize(matches.size());
    queryPoints.resize(matches.size());
    for (size_t i = 0; i < matches.size(); ++i) {
        trainPoints[i] = trainKeypoints[matches[i].trainIdx].pt;
        queryPoints[i] = queryKeypoints[matches[i].queryIdx].pt;
    }

    cv::Mat output;
    cv::Mat M = cv::findHomography(queryPoints, trainPoints, cv::RANSAC, projection_threshold);

    projectedGaze.resize(queryGaze.size());
    for (size_t i = 0; i < queryGaze.size(); ++i) {
        const double x = queryGaze[i].x * image2.rows;
        const double y = queryGaze[i].y * image2.cols;

        double k = (M.at<double>(2, 0) * x + M.at<double>(2, 1) * y + M.at<double>(2, 2));
        projectedGaze[i] = GazePoint {
            static_cast<float>((M.at<double>(0, 0) * x + M.at<double>(0, 1) * y + M.at<double>(0, 2)) / (k * image2.rows)),
            static_cast<float>((M.at<double>(1, 0) * x + M.at<double>(1, 1) * y + M.at<double>(1, 2)) / (k * image2.cols)),
        };
    }

    cv::warpPerspective(image2, output, M, cv::Size(image1.cols, image1.rows));
    for (size_t i = 0; i < trainGaze.size(); ++i) {
        cv::circle(image1,
            cv::Point2i(
                static_cast<int>(trainGaze[i].x * image1.cols), static_cast<int>(trainGaze[i].y * image1.rows)
            ), 5, cv::Scalar(0, 255, 255, 255)
        );
        cv::circle(image1,
            cv::Point2i(
                static_cast<int>(projectedGaze[i].x * image1.cols), static_cast<int>(projectedGaze[i].y * image1.rows)
            ), 5, cv::Scalar(0, 255, 255, 255)
        );
    }

    for (size_t i = 0; i < projectedGaze.size(); ++i) {
        cv::circle(output, cv::Point2i(
            static_cast<int>(projectedGaze[i].x * output.cols),
            static_cast<int>(projectedGaze[i].y * output.rows)
        ), 5, cv::Scalar(255, 255, 0, 255));
    }

    for (size_t i = 0; i < trainGaze.size(); ++i) {
        cv::circle(image2_2, cv::Point2i(
            static_cast<int>(queryGaze[i].x * image2.cols),
            static_cast<int>(queryGaze[i].y * image2.rows)
        ), 5, cv::Scalar(255, 0, 255, 255));
        //cv::circle(image2, cv::Point2i(projectedGaze[i].x * image2.cols, projectedGaze[i].y * image2.rows), 5, cv::Scalar(255, 0, 255, 255));
    }
    return output;
}

void alignImage(cv::Size2i imageSize, const std::vector<cv::KeyPoint>& trainKeypoints, const std::vector<GazePoint>& trainGaze,
                const std::vector<cv::KeyPoint>& queryKeypoints, const std::vector<GazePoint>& queryGaze,
                const std::vector<cv::DMatch>& matches,
                std::vector<GazePoint>& projectedGaze,
                const float projection_threshold,
                const int min_match_count) {
    if (!alignable(matches, min_match_count))
        return;

    std::vector<cv::Point2f> trainPoints, queryPoints;
    trainPoints.resize(matches.size());
    queryPoints.resize(matches.size());
    for (size_t i = 0; i < matches.size(); ++i) {
        trainPoints[i] = trainKeypoints[matches[i].trainIdx].pt;
        queryPoints[i] = queryKeypoints[matches[i].queryIdx].pt;
    }

    const cv::Mat M = cv::findHomography(queryPoints, trainPoints, cv::RANSAC, projection_threshold);
    if (M.empty())
        return;

    const size_t size = projectedGaze.size();
    projectedGaze.resize(size + queryGaze.size());
    for (size_t i = 0; i < queryGaze.size(); ++i) {
        const double x = queryGaze[i].x * imageSize.width;
        const double y = queryGaze[i].y * imageSize.height;

        const double k = (M.at<double>(2, 0) * x + M.at<double>(2, 1) * y + M.at<double>(2, 2));
        projectedGaze[size + i] = GazePoint {
            static_cast<float>((M.at<double>(0, 0) * x + M.at<double>(0, 1) * y + M.at<double>(0, 2)) / (k * imageSize.width)),
            static_cast<float>((M.at<double>(1, 0) * x + M.at<double>(1, 1) * y + M.at<double>(1, 2)) / (k * imageSize.height)),
        };
    }
}

cv::Mat computeSaliency(const cv::Size2i& size, double sigma, const std::vector<GazePoint>& projectedGaze, bool blur, int suppression_size, float supression) {
    cv::Mat image   = cv::Mat::zeros(size.width, size.height, CV_32F);
    cv::Mat blurred = cv::Mat::zeros(size.width, size.height, CV_32F);
    cv::Mat output  = cv::Mat::zeros(size.width, size.height, CV_8U);

    const int width  = size.width - 1;
    const int height = size.height - 1;
    float max_fixations = 0;
    for (const auto& p : projectedGaze) {
        const float fy = p.x * size.height;
        const float fx = p.y * size.width;

        //const int x = std::min(std::max(static_cast<int>(fx), 0), height);
        //const int y = std::min(std::max(static_cast<int>(fy), 0), width);
        const int x = static_cast<int>(fx);
        const int y = static_cast<int>(fy);

        if (x >= 0 && x < height && y >= 0 && y < width) {
            image.at<float>(x, y) += 1;
            max_fixations = std::max(max_fixations, image.at<float>(x, y));
        }
    }

    if (max_fixations > 1 && supression > 0 && suppression_size > 0) {
        cv::Mat kernel(suppression_size, suppression_size, CV_32F);
        kernel = 1;

        cv::Mat activations, mask, binmask;

#if defined(DEBUG_OUTPUT)
        cv::Mat test;
        image.convertTo(test, CV_8U, 255.0 / max_fixations);
        cv::imwrite("Fixations.png", test);
#endif

        filter2D(image, activations, -1, kernel, cv::Point( -1, -1 ), 0, cv::BORDER_CONSTANT);
        cv::threshold(activations, mask, supression, 255, cv::THRESH_BINARY);

#if defined(DEBUG_OUTPUT)
        activations.convertTo(test, CV_8U, 255.0 / max_fixations);
        cv::imwrite("Activations.png", test);

        mask.convertTo(binmask, CV_8U, 1.0);
        cv::imwrite("Threshold.png", binmask);
#endif

        cv::Mat outputImage = image.clone();
        image = 0;

        outputImage.copyTo(image, binmask);
    }

    if (blur) {
        cv::GaussianBlur(image, blurred, cv::Size2i(0, 0), sigma, sigma);
    } else {
        blurred = image;
    }
#ifdef PROBABILITY_MAP
    blurred = blurred / static_cast<float>(sum) * 255.0f;
#else
    double min, max;
    cv::minMaxLoc(blurred, &min, &max, 0, 0);
    blurred = blurred / max * 255.0f;
#endif

    blurred.convertTo(output, CV_8U);
    return output;
}

template <bool assignNewKeypoints>
cv::Mat filterKeyPoints(const cv::Size2i& size, std::vector<cv::KeyPoint>& kp, const cv::Mat descriptors,
                        const std::vector<GazePoint>& gaze, const cv::Size2i& patchSize,
                        const int MIN_MATCH_COUNT) {
    if (patchSize.height == 0 || patchSize.width == 0)
        return descriptors;

    cv::Mat outputDescriptors;
    std::vector<cv::KeyPoint> keypoints;
    cv::Size2f focal_patch = patchSize;
    const size_t num_keypoints = kp.size();

    // gradually incraese the focal patch if there are to few descriptors.
    do {
        const cv::Size2f fpatchSize(focal_patch.width / 2.0f, focal_patch.height / 2.0f);
        outputDescriptors = cv::Mat();
        keypoints.clear();

        for (size_t counter = 0; counter < num_keypoints; ++counter) {
            bool inside = true;
            const auto keypoint = kp[counter];
            for (const auto& gazePoint : gaze) {
                const cv::Point2f point(gazePoint.x * size.height, gazePoint.y * size.width);
                const bool point_inside = point.x >= keypoint.pt.x - fpatchSize.height &&
                                          point.x <= keypoint.pt.x + fpatchSize.height &&
                                          point.y >= keypoint.pt.y - fpatchSize.width  &&
                                          point.y <= keypoint.pt.y + fpatchSize.width;
                inside = inside && point_inside;
            }
            if (inside) {
                keypoints.emplace_back(keypoint);
                outputDescriptors.push_back(descriptors.row(static_cast<int>(counter)));
            }
        }
    
        focal_patch.height *= 2;
        focal_patch.width  *= 2;
    } while (outputDescriptors.rows < MIN_MATCH_COUNT && focal_patch.height <= 2 * size.height && focal_patch.width < 2 * size.width);

    if constexpr (assignNewKeypoints) {
        kp = keypoints;
    }

    return outputDescriptors;
}

Fixation::Fixation(Catalog& catalog, VocabularyTree& tree)
    : catalog(catalog), tree(tree) {
}

void Fixation::computeFixation(const std::string& path, uint32_t videoIndex, boost::property_tree::ptree const& config, EvaluationMethod cross_validate) {
    thread_pool& pool = thread_pool::instance();

    const cv::Size2i patchSize(config.get_child("patch_size").get<int>("x"),
                               config.get_child("patch_size").get<int>("y"));
    const size_t MAX_MATCH_COUNT = config.get<size_t>("max_match_count");
    const size_t MIN_MATCH_COUNT = config.get<size_t>("min_match_count");
    const int k_nearest = config.get<int>("k-nearest");
    const float ratio = config.get<float>("filter_ratio");
    const float sigma = config.get<float>("sigma");
    const float projection_threshold = config.get<float>("projection_threshold");
    const float supression = config.get<float>("supression_threshold");
    const int suppression_size = config.get<int>("suppression_size");
    const int optimize_gaze = config.get<int>("optimize_gaze");
    const int optimize_steps = config.get<int>("optimize_steps");
    const bool search_with_full_frame = config.get<bool>("search_with_full_train_frame");
    const bool restrict_train_frame_to_foveal_patch = config.get<bool>("restrict_train_frame_to_foveal_patch");
    const bool restrict_query_frame_to_foveal_patch = config.get<bool>("restrict_query_frame_to_foveal_patch");

    int norm = cv::NORM_L2;
    const std::string norm_type = config.get<std::string>("norm");
    if (norm_type == "L1") {
        norm = cv::NORM_L1;
    }
    else if (norm_type == "L2") {
        norm = cv::NORM_L2;
    }
    else if (norm_type == "Hamming") {
        norm = cv::NORM_HAMMING;
    }
    else if (norm_type == "Hamming2") {
        norm = cv::NORM_HAMMING2;
    } else {
        std::cout << "Unknown Norm: " << norm << std::endl;
        std::exit(-1);
    }

    auto videoPaths = catalog.getVideoPaths();
    cv::VideoCapture trainCap(videoPaths[videoIndex]);
    cv::Mat trainImage;
    trainCap.read(trainImage);

    const cv::Size2i imageSize = cv::Size2i(trainImage.rows, trainImage.cols);
    const int extention = static_cast<int>(trainCap.get(cv::CAP_PROP_FOURCC));
    const double fps =  trainCap.get(cv::CAP_PROP_FPS);
    trainCap.release();

    std::vector<size_t> videoLengths;
    videoLengths.resize(videoPaths.size());
    for (size_t i = 0; i < videoPaths.size(); ++i) {
        auto info = catalog.getVideoInfo(videoIndex);
        videoLengths[i] = info[info.size() - 1].frame;
        info.clear();
    }

    auto info = catalog.getVideoInfo(videoIndex);
    const auto frameCount = std::accumulate(std::begin(info), std::end(info), 0UL, [](size_t a, auto& inf) { return std::max(a, (size_t)inf.frame); });
    info.clear();

    cv::Mat fqueryFrame;
    std::vector<std::vector<cv::Mat>> compactQueries;
    std::vector<std::vector<std::vector<GazePoint>>> compactGaze;
    std::vector<std::vector<std::vector<cv::KeyPoint>>> compactKeypoints;
    std::vector<std::vector<GazePoint>> full_gaze;
    std::vector<std::future<bool>> done;
    full_gaze.resize(frameCount);
    done.reserve(frameCount);

    std::atomic<size_t> counter = 0;
    constexpr size_t proc_count = 1;
    const size_t output_size = proc_count * pool.threadCount();


    const size_t total = frameCount + frameCount % output_size;
    std::cout << "Compute Video " << videoIndex << std::endl;
    std::cout << "Fetch initial fixations\n";
    for (size_t i = 0; i < frameCount; i += output_size) {
        const size_t count = static_cast<size_t>(std::ceil(static_cast<double>(output_size) / pool.threadCount()));
        for (size_t t = 0; t < pool.threadCount(); ++t) {
            const size_t start = t * count + i;
            const size_t end   = std::min((t + 1) * count + i, (size_t)frameCount);

            done.push_back(pool.enqueue_task([this, videoIndex, &full_gaze, &counter, &imageSize, start, end, frameCount, patchSize, MIN_MATCH_COUNT, MAX_MATCH_COUNT,
                                                     k_nearest, ratio, projection_threshold, cross_validate, search_with_full_frame,
                                                     restrict_train_frame_to_foveal_patch, restrict_query_frame_to_foveal_patch] () {
                if (start >= frameCount || end > frameCount)
                    return false;

                cv::Mat ftrainFrame, fqueryFrame;  
                std::vector<Matching> matches;
                std::vector<GazePoint> projected;
                std::vector<cv::DMatch> filteredMatches;
                std::vector<std::vector<cv::DMatch>> discriptorMatches;
                const auto matcher = cv::FlannBasedMatcher::create();

                //for (size_t i = start; i < end; ++i) {
                const auto trainIndex = VideoIndex(videoIndex, static_cast<uint32_t>(start), -1);
                const auto trainGaze = catalog.getGaze(trainIndex);
                auto trainKeypoints = catalog.getImageKeyPoints(trainIndex);
                cv::Mat trainFrame = catalog.getImageDescriptors(trainIndex);

                auto searchTrainFrame = trainFrame;
                if (!search_with_full_frame) {
                    if (restrict_train_frame_to_foveal_patch) {
                        searchTrainFrame = filterKeyPoints<true>(imageSize, trainKeypoints, trainFrame, trainGaze, patchSize, static_cast<int>(MIN_MATCH_COUNT));
                        trainFrame = searchTrainFrame;
                    } else {
                        searchTrainFrame = filterKeyPoints<false>(imageSize, trainKeypoints, trainFrame, trainGaze, patchSize, static_cast<int>(MIN_MATCH_COUNT));
                    }
                }

                if (trainKeypoints.empty() || searchTrainFrame.empty()) {
                    if (trainKeypoints.empty() && searchTrainFrame.empty()) {
                        // This is really unfortunated as it seems that the frame has no descriptors and no keypoints.
                        // So we just ignore it for now :/
                        std::cout << "What a bummer" << std::endl;
                        return false;
                    } else if (searchTrainFrame.empty()) {
                        // We didn't find valid descriptors around the image so lets use just all descriptors then.
                        searchTrainFrame = trainFrame;
                    }
                }

                searchTrainFrame.convertTo(ftrainFrame, CV_32F);
                tree.query(ftrainFrame, matches);

                trainFrame.convertTo(ftrainFrame, CV_32F);
                matcher->clear();
                matcher->add(ftrainFrame);
                matcher->train();

                if (cross_validate == EvaluationMethod::CrossValidation) {
                    matches.erase(std::remove_if(std::begin(matches), std::end(matches), [videoIndex](const auto& elem) {
                        return elem.video == videoIndex;
                    }), std::end(matches));
                }
                else if (cross_validate == EvaluationMethod::TemporalCrossValidation) {
                    matches.erase(std::remove_if(std::begin(matches), std::end(matches), [videoIndex, start](const auto& elem) {
                        return elem.video == videoIndex && elem.frame == start;
                    }), std::end(matches));
                }
                else if (cross_validate == EvaluationMethod::PastTemporalCrossValidation) {
                    matches.erase(std::remove_if(std::begin(matches), std::end(matches), [videoIndex, start](const auto& elem) {
                        return elem.video == videoIndex && elem.frame >= start;
                    }), std::end(matches));
                }

                matches.erase(std::begin(matches) + std::min(MAX_MATCH_COUNT, matches.size() - 1), std::end(matches));
                std::sort(std::begin(matches), std::end(matches), [](const auto& a, const auto& b) {
                    return a.video == b.video ? a.frame < b.frame : a.video < b.video;
                });

#if defined(DEBUG_OUTPUT) && false
{
                cv::Mat initFrame;
                cv::VideoCapture cap(catalog.getVideoPath(trainIndex.video));
                size_t f = 0;
                do { cap.read(initFrame); } while (f++ < trainIndex.frame);
                cv::imwrite("Initial.png", initFrame);

                size_t current_index = 0;
                while (current_index < MAX_MATCH_COUNT) {
                    cv::Mat mat;
                    auto queryIndex = VideoIndex(matches[current_index].video, matches[current_index].frame, -1);
                    auto queryKeypoints = catalog.getImageKeyPoints(queryIndex);
                    cv::VideoCapture cap(catalog.getVideoPath(matches[current_index].video));
                    size_t f = 0;
                    do { cap.read(mat); } while (f++ < matches[current_index].frame);

                    //cv::Mat queryFrame  = catalog.getImageDescriptors(queryIndex);
                    //matcher->match(trainFrame, queryFrame, filteredMatches, cv::noArray()); // knnMatch(trainFrame, queryFrame, discriptorMatches, 1);
                    //filterMatches(discriptorMatches, filteredMatches, 0.75);

                    //cv::Mat img;
                    //cv::drawMatches(initFrame, trainKeypoints, mat, queryKeypoints, filteredMatches, img);

                    cv::imwrite("MatchPatch-" + std::to_string(current_index) + ".png", mat);
                });
}
#endif

#if defined(DEBUG_OUTPUT)
                cv::Mat initFrame;
                cv::VideoCapture cap(catalog.getVideoPath(trainIndex.video));
                size_t f = 0;
                do { cap.read(initFrame); } while (f++ < trainIndex.frame);
                cv::putText(initFrame, "Train Frame", cv::Point2i(10, 80), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255));
#endif

                if (restrict_query_frame_to_foveal_patch) {
                    for (const Matching& match : matches) {
                        const auto queryIndex = VideoIndex(match.video, match.frame, -1);
                        const auto queryGaze = catalog.getGaze(queryIndex);
                        if (queryGaze.empty()) continue;
                        auto queryKeypoints = catalog.getImageKeyPoints(queryIndex);
                        cv::Mat queryFrame  = catalog.getImageDescriptors(queryIndex);
                        queryFrame = filterKeyPoints<true>(imageSize, queryKeypoints, queryFrame, queryGaze, patchSize, static_cast<int>(MIN_MATCH_COUNT));

                        queryFrame.convertTo(fqueryFrame, CV_32F);
                        matcher->knnMatch(fqueryFrame, discriptorMatches, k_nearest);
                        filterMatches(discriptorMatches, filteredMatches, ratio);

#if defined(DEBUG_OUTPUT)
                            cv::Mat mat;
                            cv::VideoCapture cap(catalog.getVideoPath(queryIndex.video));
                            size_t f = 0;
                            do { cap.read(mat); } while (f++ < queryIndex.frame);
                            cv::Mat projection = mat.clone();
                            cv::putText(mat, "Query Frame", cv::Point2i(10, 80), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255));

                            cv::Mat img;
                            cv::drawMatches(mat, queryKeypoints, initFrame, trainKeypoints, filteredMatches, img);
                            cv::imwrite("MatchPatch-" + std::to_string(current_index) + ".png", img);
#endif

                        alignImage(imageSize,
                                    trainKeypoints, trainGaze,
                                    queryKeypoints, queryGaze,
                                    filteredMatches, projected, projection_threshold, static_cast<int>(MIN_MATCH_COUNT));

#if defined(DEBUG_OUTPUT)
                        auto tst = alignImage(initFrame, trainKeypoints, trainGaze,
                                            mat, projection, queryKeypoints, queryGaze,
                                            filteredMatches, projected,
                                            projection_threshold, MIN_MATCH_COUNT);
                        if (!tst.empty()) {
                            cv::putText(tst, "Projected Frame", cv::Point2i(10, 80), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255));

                            cv::Mat tst2, tst3;
                            cv::hconcat(initFrame, tst, tst2);
                            cv::hconcat(mat, tst2, tst3);
                            cv::imwrite("Reproject-" + std::to_string(current_index) + ".png", tst3);
                        }
#endif

                        filteredMatches.clear();
                        discriptorMatches.clear();
                    } 
                } else {
                    for (const Matching& match : matches) {
                        const auto queryIndex = VideoIndex(match.video, match.frame, -1);
                        auto queryGaze = catalog.getGaze(queryIndex);
                        auto queryKeypoints = catalog.getImageKeyPoints(queryIndex);
                        cv::Mat queryFrame  = catalog.getImageDescriptors(queryIndex);

                        queryFrame.convertTo(fqueryFrame, CV_32F);
                        matcher->knnMatch(fqueryFrame, discriptorMatches, k_nearest);
                        filterMatches(discriptorMatches, filteredMatches, ratio);

                        alignImage(imageSize, trainKeypoints, trainGaze, queryKeypoints, queryGaze, filteredMatches, projected, projection_threshold, static_cast<int>(MIN_MATCH_COUNT));
                        
                        filteredMatches.clear();
                        discriptorMatches.clear();
                    }
                }

                full_gaze[start] = projected;

#if defined(DEBUG_OUTPUT)
                cv::Mat initFrame;
                cv::VideoCapture cap(catalog.getVideoPath(trainIndex.video));
                size_t f = 0;
                do { cap.read(initFrame); } while (f++ < trainIndex.frame);
                cv::putText(initFrame, "Train Frame", cv::Point2i(10, 80), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255));

                cv::Mat sal, out;
                cv::cvtColor(output[i - offset], sal, cv::COLOR_GRAY2BGR);
                cv::addWeighted(initFrame, 0.5, sal, 0.5, 0.0, out);
                cv::imwrite("Patched.png", out);
#endif
                    
                projected.clear();
                std::cout << "\r" + std::to_string(++counter) + "/" + std::to_string(frameCount) << std::flush;
                return true;
            }));
        }
    }

    for (size_t t = 0; t < frameCount; ++t) {
        done[t].wait();
    }
    done.clear();

    if (optimize_gaze > 0) {
        std::cout << "\nOptimize gaze\n";
        
        for (int k = 0; k < optimize_steps; ++k) {
            const auto old_full_gaze = full_gaze;
            for (size_t i = 0, counter = 0; i < frameCount; i += output_size) {
                const size_t count = static_cast<size_t>(std::ceil(static_cast<double>(output_size) / pool.threadCount()));
                for (size_t t = 0; t < pool.threadCount(); ++t) {
                    const size_t offset = i;
                    const size_t start = t * count + i;
                    const size_t end   = std::min((t + 1) * count + i, (size_t)frameCount);
                
                    done.push_back(pool.enqueue_task([this, &counter, videoIndex, start, end, frameCount, &old_full_gaze, &full_gaze, MAX_MATCH_COUNT, MIN_MATCH_COUNT, k_nearest, ratio, imageSize, projection_threshold, optimize_gaze, cross_validate]() {
                        if (start >= frameCount || end > frameCount)
                            return true;
                        
                        std::vector<Matching> matches;
                        std::vector<cv::DMatch> filteredMatches;
                        std::vector<std::vector<cv::DMatch>> discriptorMatches;
                        const auto matcher = cv::FlannBasedMatcher::create();
                        cv::Mat ftrainFrame, fqueryFrame;

                        for (size_t i = start; i < end; ++i) {
                            const auto trainIndex = VideoIndex(videoIndex, static_cast<uint32_t>(i), -1);
                            const auto trainKeypoints = catalog.getImageKeyPoints(trainIndex);
                            const cv::Mat trainFrame = catalog.getImageDescriptors(trainIndex);
                            if (trainFrame.empty()) continue;

                            matches.clear();
                            trainFrame.convertTo(ftrainFrame, CV_32F);
                            tree.query(ftrainFrame, matches);
                    
                            matcher->clear();
                            matcher->add(ftrainFrame);
                            matcher->train();
                            
                            // only search inside the video
                            matches.erase(std::remove_if(std::begin(matches), std::end(matches), [videoIndex](const auto& elem) {
                                return elem.video != videoIndex;
                            }), matches.end());
                            if (cross_validate == EvaluationMethod::PastTemporalCrossValidation) {
                                matches.erase(std::remove_if(std::begin(matches), std::end(matches), [i](const auto& elem) {
                                    return elem.frame >= i;
                                }), std::end(matches));
                            }

                            for (size_t j = 0; j < std::min((size_t)optimize_gaze, matches.size()); ++j) {
                                const auto& match = matches[j];
                                if (match.frame == i || match.frame < 0 || match.frame >= frameCount) {
                                    continue;
                                }

                                const auto queryIndex = VideoIndex(videoIndex, match.frame, -1);
                                const auto queryKeypoints = catalog.getImageKeyPoints(queryIndex);
                                cv::Mat queryFrame = catalog.getImageDescriptors(queryIndex);
                                if (queryFrame.empty()) continue;

                                queryFrame.convertTo(fqueryFrame, CV_32F);
                                matcher->knnMatch(fqueryFrame, discriptorMatches, k_nearest);
                                
                                filterMatches(discriptorMatches, filteredMatches, ratio);
                                alignImage(imageSize, trainKeypoints, old_full_gaze[i], queryKeypoints, old_full_gaze[match.frame], filteredMatches, full_gaze[i], projection_threshold, static_cast<int>(MIN_MATCH_COUNT));

                                filteredMatches.clear();
                                discriptorMatches.clear();
                            }

                            std::string out_str = "\r" + std::to_string(++counter) + "/" + std::to_string(full_gaze.size());
                            std::cout << out_str << std::flush;
                            matches.clear();
                        }

                        return true;
                    }));
                }
            }

            for (long i = 0; i < done.size(); ++i) {
                done[i].wait();
            }
            done.clear();
        }
    }

    std::vector<cv::Mat> output;
    output.resize(output_size);
    done.clear();

    cv::VideoWriter writer = cv::VideoWriter(path, extention, fps, cv::Size2i(trainImage.cols, trainImage.rows), false);
    std::cout << "\nCompute saliency maps from fixations\n";
    counter = 0;
    for (size_t i = 0; i < frameCount; i += output_size) {
        const size_t count = static_cast<size_t>(std::ceil(static_cast<double>(output_size) / pool.threadCount()));
        for (size_t t = 0; t < pool.threadCount(); ++t) {
            const size_t offset = i;
            const size_t start = t * count + i;
            const size_t end   = std::min((t + 1) * count + i, (size_t)frameCount);

            done.push_back(pool.enqueue_task([this, &counter, &full_gaze, &output, &imageSize, start, end, offset, frameCount, sigma, suppression_size, supression]() {
                for (size_t i = start; i < end; ++i) {
                    if (start >= frameCount || end >= frameCount) {
                        output[i - offset] = cv::Mat::zeros(imageSize.width, imageSize.height, CV_8U);
                        return true;
                    }

                    output[i - offset] = computeSaliency(imageSize, sigma, full_gaze[i], sigma > 0, suppression_size, supression);

                    std::string out_str = "\r" + std::to_string(++counter) + "/" + std::to_string(full_gaze.size());
                    std::cout << out_str << std::flush;
                }

                return true;
            }));
        }

        for (size_t t = 0; t < done.size(); ++t) {
            done[t].wait();
            
            for (size_t j = 0; j < proc_count; ++j) {
                if (i + j + t * proc_count < frameCount) {
                    writer.write(output[j + t * proc_count]);
                    output[j + t * proc_count].release();
                }
            }
        }
        done.clear();
    } 

    writer.release();
}
