
#ifndef TYPES_H
#define TYPES_H

#include <string>
#include <vector>

#include <cv.h>
#include <highgui.h>

using namespace std;

typedef unsigned char uchar;
typedef unsigned int uint;

typedef cv::Mat_<cv::Vec3b> InputFrame;
typedef cv::MatConstIterator_<cv::Vec3b> InputIterator;

typedef cv::Mat_<cv::Vec3d> FPMatrix;
typedef cv::MatIterator_<cv::Vec3d> FPIterator;
typedef cv::MatConstIterator_<cv::Vec3d> FPConstIterator;

typedef cv::Mat_<uchar> BinaryFrame;
typedef cv::MatIterator_<uchar> BinaryIterator;
typedef cv::MatConstIterator_<uchar> BinaryConstIterator;

typedef BinaryFrame GTFrame;
typedef BinaryConstIterator GTIterator;

typedef vector<string> Arguments;

template <class T>
T lexical_cast(const string& textNumber) {
	istringstream ss(textNumber);

	T number;
	ss >> number;

	return number;
}

const uchar BLACK = 0;
const uchar SHADOW = 50;
const uchar OUTOFSCOPE = 85;
const uchar UNKNOWN = 170;
const uchar WHITE = 255;

#endif