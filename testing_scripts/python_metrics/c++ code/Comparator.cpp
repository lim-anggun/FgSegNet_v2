
#include "Comparator.h"

#include <fstream>

Comparator::Comparator(const VideoFolder &videoFolder)
	: videoFolder(videoFolder),
	ROI(cv::imread(videoFolder.getVideoPath() + "ROI.bmp", 0))
{}

void Comparator::compare() {
	const Range range = videoFolder.getRange();
	const uint fromIdx = range.first;
	const uint toIdx = range.second;

	tp = fp = fn = tn = 0;
	nbShadowErrors = 0;
	cv::Mat gt;
	cv::Mat pred;
	// For each frame in the range, compare and calculate the statistics
	for (uint t = fromIdx; t <= toIdx; ++t) {
		if(std::ifstream(videoFolder.binaryFrame(t)) && std::ifstream(videoFolder.gtFrame(t))){
			pred = cv::imread(videoFolder.binaryFrame(t), 0);
			gt = cv::imread(videoFolder.gtFrame(t), 0);
			compare(pred, gt);
		}
	}
	printf(" %d, %d, %d, %d, %d\n",tp,fp,fn,tn,nbShadowErrors);
}

void Comparator::compare(const BinaryFrame& binary, const GTFrame& gt) {
	if (binary.empty()) {
		throw string("Binary frame is null. Probably a bad path or incomplete folder.\n");
	}
	
	if (gt.empty()) {
		throw string("gt frame is null. Probably a bad path or incomplete folder.\n");
	}
	
	BinaryConstIterator itBinary = binary.begin();
	GTIterator itGT = gt.begin();
	ROIIterator itROI = ROI.begin();

	BinaryConstIterator itEnd = binary.end();
	for (; itBinary != itEnd; ++itBinary, ++itGT, ++itROI) {
		// Current pixel needs to be in the ROI && it must not be an unknown color
		if (*itROI != BLACK && *itGT != UNKNOWN) {

			if (*itBinary == WHITE) { // Model thinks pixel is foreground
				if (*itGT == WHITE) {
					++tp; // and it is
				} else {
					++fp; // but it's not
				}
			} else { // Model thinks pixel is background
				if (*itGT == WHITE) {
					++fn; // but it's not
				} else {
					++tn; // and it is
				}
			}

			if (*itGT == SHADOW) {
				if (*itBinary == WHITE) {
					++nbShadowErrors;
				}
			}

		}
	}
}

void Comparator::save() const {
	const string filePath = videoFolder.getVideoPath() + "stats.txt";
	ofstream f(filePath.c_str(), ios::out | ios::app);
	if (f.is_open()) {
		f << "cm: " << tp << ' ' << fp << ' ' << fn << ' ' << tn << ' ' << nbShadowErrors;
		f.close();
	} else {
		throw string("Unable to open the file : ") + filePath;
	}
}
