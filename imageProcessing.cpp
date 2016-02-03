#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "imageProcessing.h"
#include <dirent.h>

using namespace std;
using namespace cv;

Mat loadGrayScaleImage(string imageFname) {
    Mat image;
    image = imread(imageFname, CV_LOAD_IMAGE_GRAYSCALE);

    if (!image.data) {
        throw runtime_error("Could not open or find the image");
    }

    return image;
}

void showImage(Mat image) {
    namedWindow("Display window", WINDOW_AUTOSIZE);
    imshow("Display window", image);

    waitKey(0);
    return;
}

vector<double> readImage(string imageFname) {
    Mat image = loadGrayScaleImage(imageFname);
    vector<double> dataElem;

    dataElem.assign(image.datastart, image.dataend);

    return dataElem;
}

vector<vector<double>> readImagesDir(string dirName) {
    vector<vector<double>> data;

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(dirName.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            string imageFname = dirName + ent->d_name;
            try {
                data.push_back(readImage(imageFname));
            } catch (runtime_error err) {
                cout << err.what() << endl;
            }
        }
        closedir(dir);
    }

    return data;
}
