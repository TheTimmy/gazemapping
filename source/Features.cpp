#include "features.h"

#include <boost/filesystem.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/hdf.hpp>

// #include <sys/ioctl.h> //ioctl() and TIOCGWINSZ
// #include <unistd.h> // for STDOUT_FILENO

#include <mutex>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "Pool.h"
#include "MatPersistor.h"
#include "ShootSegmenter.h"
#include "Catalog.h"
#include "VocabularyTree.h"
#include "KeyPointPersistor.h"
#include "VecKeyPointPersistor.h"
#include "Fixation.h"

#include <boost/property_tree/ptree.hpp>

template <typename T, typename... Args>
void compute_descriptor(boost::filesystem::path vocabPath,
                        const std::string filename,
                        const size_t videoIndex,
                        boost::property_tree::ptree const& config,
                        Args... args) {

    cv::VideoCapture cap { filename };
    if (!cap.isOpened()) {
        cap.release();
        std::cout << "Video " << filename << " does not exist" << std::endl;
        return;
    }

    const int MIN_KEYPOINTS = config.get<int>("min_keypoints");
    std::string output_path = config.get<std::string>("output_directory");
    if (output_path.empty()) {
        const auto path = boost::filesystem::path(filename);
        output_path = boost::filesystem::change_extension(path, "").string();
    }

    if (boost::filesystem::exists(boost::filesystem::path(output_path + ".vid")) &&
        boost::filesystem::exists(boost::filesystem::path(output_path + ".key"))) {
        return;
        VecMatPersistor persistor1 = VecMatPersistor::open(output_path + ".vid");
        VecKeyPointPersistor persistor2 = VecKeyPointPersistor::open(output_path + ".key");

        const int frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        const size_t size1 = persistor1.getDescriptorInfo().size();
        const size_t size2 = persistor2.getDescriptorInfo().size();
        persistor1.close();
        persistor2.close();

        std::cout << "frames " << frames << " s1: " << size1 << " s2 " << size2 << std::endl;
        if (size1 == frames && size2 == frames) {
            std::cout << "Skip Video: " << filename << std::endl;
            return;
        }
    }
    
    bool hasFrame = true;
    bool visualize = config.get<bool>("visualize");
    cv::Mat frame, descriptor, uintDescriptor;
    std::vector<std::vector<cv::KeyPoint>> keypoints;
    std::vector<cv::Mat> frames;
    std::vector<uint32_t> frameIndices;

    cv::Ptr<T> sift = T::create(args...);
    auto frameCount = std::to_string((size_t)cap.get(cv::CAP_PROP_FRAME_COUNT));

    size_t frameIndex = 0;
    cv::VideoWriter writer;
    if(visualize) {
        Size S = Size((int) cap.get(CAP_PROP_FRAME_WIDTH),    // Acquire input size
                      (int) cap.get(CAP_PROP_FRAME_HEIGHT));
        int ex = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));

        std::cout << "Write to: " << boost::filesystem::path(filename).stem().string() + "-features" + boost::filesystem::path(filename).extension().string() << std::endl;
        writer = cv::VideoWriter(boost::filesystem::path(filename).stem().string() + "-features" + boost::filesystem::path(filename).extension().string(),
                                 ex, cap.get(CAP_PROP_FPS), S, true);
    }

    while(true) {
        const std::string index = std::to_string(frameIndex);
        const std::string process = "\033[" + std::to_string(videoIndex + 1) + ";" + "0" + "H" + filename +  ": " + index + '/' + frameCount;
        std::cout << process;

        cap >> frame;
        if (frame.empty())
            break;

        std::vector<KeyPoint> keps;
        sift->detectAndCompute(frame, cv::noArray(), keps, descriptor, false);
        if (keps.size() >= MIN_KEYPOINTS) {
            descriptor.convertTo(uintDescriptor, CV_8U);

            frameIndices.emplace_back(frameIndex);
            frames.emplace_back(uintDescriptor.clone());
            keypoints.emplace_back(keps);
        }

        if (visualize) {
            cv::drawKeypoints(frame, keps, frame, cv::Scalar_<double>::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            writer.write(frame);
        }

        ++frameIndex;
    }

    if(visualize) {
        writer.release();
    }


    VecMatPersistor::create(output_path + ".vid", frames, frameIndices);
    VecKeyPointPersistor::create(output_path + ".key", keypoints, frameIndices);
    cap.release();
}

void compute_descriptors(const Catalog& catalog, boost::property_tree::ptree const& config) {
    thread_pool& pool = thread_pool::construct(8);
    auto output = catalog.getVideoStorage();
    auto paths = catalog.getVideoPaths();

    // TODO Stupid hack to work around the program not processing the last batch of videos
    // I really dont know why this is happening tho :/
    for (int i = 0; i < 8; ++i) {
        output.push_back(output[i]);
        paths.push_back(paths[i]);
    }

    auto extractor = config.get_child("extractor");
    if (extractor.get<std::string>("type") == "SIFT") {
        int nFeatures = extractor.get<int>("nfeatures");
        int nOctaveLayers = extractor.get<int>("nOctaveLayers");
        double contrastThreshold = extractor.get<double>("contrastThreshold");
        double edgeThreshold = extractor.get<double>("edgeThreshold");
        double sigma = extractor.get<double>("sigma");

        for (size_t i = 0; i < output.size(); ++i) {
            pool.enqueue_task(compute_descriptor<cv::SIFT, int, int, double, double, double>,
                              output[i], paths[i], i, config, nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
        }
    } else {
        std::cout << "Terminating: Unkown feature extractor " << extractor.get<std::string>("type") << std::endl;
        exit(-1);
    }
}



/*void compute_fixations() {
    cv::Size2i patch_size(128, 128);
    for(int label = 1; label <= 5; ++label) {
        compute_fixation(label, patch_size);
    }
}*/
