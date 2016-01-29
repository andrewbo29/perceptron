#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "imageProcessing.h"

using namespace std;
using namespace cv;

Mat loadGrayScaleImage(string imageFname) {
    Mat image;
    image = imread(imageFname, CV_LOAD_IMAGE_GRAYSCALE);

    if (!image.data) {
        cout << "Could not open or find the image" << std::endl;
        Mat emptyImage;
        return emptyImage;
    }

    return image;
}

void showImage(Mat image) {
    namedWindow("Display window", WINDOW_AUTOSIZE);
    imshow("Display window", image);

    waitKey(0);
    return;
}
