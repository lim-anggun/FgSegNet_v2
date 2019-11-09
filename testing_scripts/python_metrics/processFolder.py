#!/usr/bin/python
# -*- coding: utf-8 -*-

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Nil Goyette
# University of Sherbrooke
# Sherbrooke, Quebec, Canada. April 2012

"""Please notice that in the metrics you calculate may different from the ones
 that are going to be shown on changedetection.net, since only half of the 
ground truth is available to calculate locally with this code, while the 
changedetection.net calculates metrics based on all the ground truth."""

import os
import shutil
import subprocess
import sys

from Stats import Stats

call = subprocess.call

def main():    
    datasetPath = sys.argv[1]
    binaryRootPath = sys.argv[2]
    
    if not isValidRootFolder(datasetPath):
        print('The folder ' + datasetPath + ' is not a valid root folder.');
        return
    
    processFolder(datasetPath, binaryRootPath)

def processFolder(datasetPath, binaryRootPath):
    """Call your executable for all sequences in all categories."""
    stats = Stats(datasetPath)  #STATS
    f = open(datasetPath + '\\' +  'fscore.txt', 'w')
    for category in getDirectories(datasetPath):
        stats.addCategories(category)  #STATS
        
        categoryPath = os.path.join(datasetPath, category)
        for video in getDirectories(categoryPath):
            videoPath = os.path.join(categoryPath, video)
            binaryPath = os.path.join(binaryRootPath, category, video)
            if isValidVideoFolder(videoPath):
                confusionMatrix = compareWithGroungtruth(videoPath, binaryPath)
                stats.update(category, video, confusionMatrix)
                alpha = 0.000001
                fscore = (2.0 * confusionMatrix[0])/ (((2.0 * confusionMatrix[0]) + confusionMatrix[1] + confusionMatrix[2]) + alpha)
                f.write(video + ' : ' + str(fscore) + '\n')
            else:
                print ('Invalid folder : ' + videoPath)
        stats.writeCategoryResult(category)
    stats.writeOverallResults()
    f.close()

def compareWithGroungtruth(videoPath, binaryPath):
    """Compare your binaries with the groundtruth and return the confusion matrix"""
    statFilePath = os.path.join(videoPath, 'stats.txt')
    deleteIfExists(statFilePath)

    groundtruthPath = os.path.join(videoPath, 'groundtruth')
    retcode = call([os.path.join('exe', 'comparator.exe'),
                    videoPath, binaryPath],
                   shell=True)
    
    return readCMFile(statFilePath)

def readCMFile(filePath):
    """Read the file, so we can compute stats for video, category and overall."""
    if not os.path.exists(filePath):
        print("The file " + filePath + " doesn't exist.\nIt means there was an error calling the comparator.")
        raise Exception('error')
    
    with open(filePath) as f:
        for line in f.readlines():
            if line.startswith('cm:'):
                numbers = line.split()[1:]
                return [int(nb) for nb in numbers[:5]]





def isValidRootFolder(path):
    """A valid root folder must have the six categories"""
    #categories = set(['Board_a', 'Candela_m1.10_a', 'CAVIAR1_a', 'CAVIAR2_a', 'CaVignal_a','Foliage_a', 'HallAndMonitor_a', 'HighwayI_a', 'HighwayII_a', 'HumanBody2_a', 'IBMtest2_a', 'PeopleAndFoliage_a', 'Toscana_a','Snellen_a'])
    categories = set(['dynamicBackground', 
                      'baseline',  
                      'cameraJitter', 
                      'intermittentObjectMotion', 
                      'shadow', 
                      'thermal', 
                      'badWeather', 
                      'lowFramerate', 
                      'nightVideos', 
                      'PTZ', 
                      'turbulence'
                      ])
    #categories = set(['baseline_2', 'baseline_4', 'thermal_2', 'thermal_4', 'cameraJitter_2', 'cameraJitter_4'])

    folders = set(getDirectories(path))
    return len(categories.intersection(folders)) == 11

def isValidVideoFolder(path):
    """A valid video folder must have \\groundtruth, \\input, ROI.bmp, temporalROI.txt"""
    return os.path.exists(os.path.join(path, 'groundtruth'))  and os.path.exists(os.path.join(path, 'ROI.bmp')) and os.path.exists(os.path.join(path, 'temporalROI.txt'))
    # and os.path.exists(os.path.join(path, 'input'))
def getDirectories(path):
    """Return a list of directories name on the specifed path"""
    return [file for file in os.listdir(path)
            if os.path.isdir(os.path.join(path, file))]

def deleteIfExists(path):
    if os.path.exists(path):
        os.remove(path)


if __name__ == "__main__":
    main()
