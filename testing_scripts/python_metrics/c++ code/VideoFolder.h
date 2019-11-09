	
#pragma once

#include <string>

#include "types.h"

typedef pair<uint, uint> Range;

class VideoFolder
{
	public:
		VideoFolder(const string& videoPath, const string& binaryPath);

		void readTemporalFile();
		void setExtension();

		const string firstFrame() const;
		const string inputFrame(const uint idx) const;
		const string binaryFrame(const uint idx) const;
		const string gtFrame(const uint idx) const;

		const Range getRange() const { return range; }
		const string getVideoPath() const { return videoPath; }

	private:
		const char sep;
		string binaryExtension;

		const string videoPath;
		const string binaryPath;
		Range range;

		const string toFrameNumber(const uint idx) const;
};

