
#pragma once

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include "types.h"
#include "VideoFolder.h"

class Comparator
{
	public:
		Comparator(const VideoFolder &videoFolder);

		void compare();
		void save() const;

	private:
		typedef BinaryFrame ROIFrame;
		typedef BinaryConstIterator ROIIterator;

		const VideoFolder &videoFolder;
		const ROIFrame ROI;

		uint tp, fp, fn, tn;
		uint nbShadowErrors;

		void compare(const BinaryFrame& binary, const GTFrame& gt);
};
