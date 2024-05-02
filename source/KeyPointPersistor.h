#pragma once

#include <stdlib.h>
#include <opencv2/core.hpp>
#include <vector>

using namespace cv;
using namespace std;

class KeyPointPersistor {
public:

    static void persist(string file_path, vector<KeyPoint> &kps);

    static void restore(string file_path, vector<KeyPoint> &kps);

    static void append(string file_path, vector<KeyPoint> &kps);

private:
    static void copyTo(Mat &aux, vector<KeyPoint> &kps);
};
