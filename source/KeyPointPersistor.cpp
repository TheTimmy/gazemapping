#include "KeyPointPersistor.h"
#include "MatPersistor.h"

#include <vector>

using namespace std;
using namespace cv;


void KeyPointPersistor::copyTo(Mat &aux, vector<KeyPoint> &kps) {
    aux.create(kps.size(), 6, CV_32F);
    for (unsigned int i = 0; i < kps.size(); i++) {

        KeyPoint kp = kps.at(i);

        aux.at<float>(i, 0) = kp.pt.x;
        aux.at<float>(i, 1) = kp.pt.y;
        aux.at<float>(i, 2) = kp.angle;
        aux.at<float>(i, 3) = kp.size;
        aux.at<float>(i, 4) = kp.octave;
        aux.at<float>(i, 5) = kp.response;
    }
}

void KeyPointPersistor::persist(string file_path, vector<KeyPoint> &kps) {
    Mat aux;
    copyTo(aux, kps);

    MatPersistor mp(file_path);
    mp.create(aux);
}

void KeyPointPersistor::append(string filePath, vector<KeyPoint> &kps) {
    Mat aux;
    copyTo(aux, kps);

    MatPersistor mp(filePath);
    if (!mp.exists()) {
        mp.create(aux);
    } else {
        mp.openWrite();
        mp.append(aux);
        mp.close();
    }
}


void KeyPointPersistor::restore(string file_path, vector<KeyPoint> &kps) {
    Mat aux;

    MatPersistor mp(file_path);
    mp.openRead();
    mp.read(aux);
    kps.clear();

    for (int i = 0; i < aux.rows; i++) {

        KeyPoint kp;
        kp.pt.x = aux.at<float>(i, 0);
        kp.pt.y = aux.at<float>(i, 1);
        kp.angle = aux.at<float>(i, 2);
        kp.size = aux.at<float>(i, 3);
        kp.octave = aux.at<float>(i, 4);
        kp.response = aux.at<float>(i, 5);

        kps.push_back(kp);
    }
}
