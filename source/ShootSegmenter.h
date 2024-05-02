#pragma once

#include <opencv2/core.hpp>

class ShootSegmenter {

    /**
     * Since a video can have many pictures almost duplicated,
     * it is interesting to have a mechanism to choose a subset of that frames.
     *
     * This class receives a sequence of frames (from a video) and
     * decides which frames to consider (trying discarding almost duplicated frames).
     *
     */

public:

    /**
     * Constructor
     */
    ShootSegmenter();

    /**
    * Destructor
    */
    virtual ~ShootSegmenter();

    /**
     * Decides if this frame has to be considered
     * @param frame input frame
     * @return true if this frame has to be considered
     */
    bool chooseThisFrame(cv::Mat &frame);


private:
    static const int SHRINK_SIZE = 16;
    static const int THRESH_DIFF = 200;
    static const int STAB_ = 800;
    static const int MAX_RESAMPLE = 12;

    int _numFrames;
    int _lastBigJum;
    int _lastChosenPos;
    cv::Mat _shf;
    cv::Mat _lshf;
    cv::Mat _lastChosenShk;

    void shrink(cv::Mat &img, int maxDim, cv::Mat &out);
};