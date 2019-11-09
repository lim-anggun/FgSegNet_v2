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

import functools
import math
import numpy as np

foldl = functools.reduce

NBSTATS = 5
(TP, FP, FN, TN, NBSHADOWERROR) = range(NBSTATS)

def addVectors(lst1, lst2):
    """Add two lists together like a vector."""
    return list(map(int.__add__, lst1, lst2))

def sumListVectors(lst):
    """Calculate the sum of a list of 4-values vector."""
    return foldl(addVectors, lst, [0, 0, 0, 0, 0])

def getStats(cm):
    """Return the usual stats for a confusion matrix."""
    TP, FP, FN, TN, SE = cm
    alpha = 0.00001
    recall = TP / (TP + FN + alpha)
    specficity = TN / (TN + FP + alpha)
    fpr = FP / (FP + TN + alpha)
    fnr = FN / (TP + FN + alpha)
    pbc = 100.0 * (FN + FP) / (TP + FP + FN + TN + alpha)
    precision = TP / ((TP + FP) + alpha)
    fmeasure = 2.0 * (recall * precision) / ((recall + precision) + alpha)
    mcc = (1.0 * ((TP * TN) - (FP * FN)))/np.sqrt((TP + FP)*(TP + FN)*(TN + FN)*(TN + FP) + alpha)
    pwc = (100.0 * (FP + FN))/ (TP + FP + TN + FN + alpha)

    return [recall, specficity, fpr, fnr, pbc, precision, fmeasure, mcc, pwc]

def mean(l):
    """Return the mean of a list."""
    return sum(l) / len(l)

def cmToText(cm):
    return ' '.join([str(val) for val in cm])

def writeComment(f):
    f.write('#This is the statistical file we use to compare each method.\n')
    f.write('#Only the lines starting with "cm" are importants.\n')
    f.write('#cm NbTruePositive NbFalsePositive NbFalseNegative NbTrueNegative NbErrorShadow\n\n')

class Stats:
    def __init__(self, path):
        self.path = path
        self.categories = dict()

    def addCategories(self, category):
        if category not in self.categories:
            self.categories[category] = {}

    def update(self, category, video, confusionMatrix):
        self.categories[category][video] = confusionMatrix

    def writeCategoryResult(self, category):
        """Write the result for each category."""
        videoList = list(self.categories[category].values())
        categoryStats = []

        categoryTotal = sumListVectors(videoList)
        with open(self.path + '\\' + category + '\\' + 'stats.txt', 'w') as f:
            writeComment(f)
            for video, cm in self.categories[category].items():
                categoryStats.append(getStats(cm))
                f.write('cm video ' + category + ' ' + video + ' ' + cmToText(cm) + '\n')
                
            f.write('cm category ' + category + ' ' + cmToText(categoryTotal) + '\n\n')
            f.write('\nRecall\t\t\tSpecificity\t\tFPR\t\t\t\tFNR\t\t\t\tPBC\t\t\t\tPrecision\t\tFMeasure\t\tMCC\t\tPWC')
            f.write('\n{0:1.10f}\t{1:1.10f}\t{2:1.10f}\t{3:1.10f}\t{4:1.10f}\t{5:1.10f}\t{6:1.10f}\t{7:1.10f}\t{8:1.10f}'.format(*[mean(z) for z in zip(*categoryStats)]))
        
    def writeOverallResults(self):
        """Write overall results."""
        totalPerCategoy = [sumListVectors(list(CMs.values())) for CMs in self.categories.values()]
        categoryStats = {}
        
        with open(self.path + '\\' + 'stats.txt', 'w') as f:
            writeComment(f)
            for category in self.categories.keys():
                videoList = list(self.categories[category].values())
                categoryStats[category] = []

                categoryTotal = sumListVectors(videoList)
                for video, cm in self.categories[category].items():
                    categoryStats[category].append(getStats(cm))
                    f.write('cm video ' + category + ' ' + video + ' ' + cmToText(cm) + '\n')
                f.write('cm category ' + category + ' ' + cmToText(categoryTotal) + '\n\n')

                cur = [mean(z) for z in zip(*categoryStats[category])]
                categoryStats[category] = cur
                

            total = sumListVectors(totalPerCategoy)
            f.write('\n\ncm overall ' + cmToText(total))

            overallStats = []
            f.write('\n\n\n\n\t\t\tRecall\t\t\tSpecificity\t\tFPR\t\t\t\tFNR\t\t\t\tPBC\t\t\t\tPrecision\t\tFMeasure\t\tMCC\t\t\t\tPWC')
            for category, stats in categoryStats.items():
                overallStats.append(stats)
                if len(category) > 10:
                    category = category[:9]
                f.write('\n{0} :\t{1:1.10f}\t{2:1.10f}\t{3:1.10f}\t{4:1.10f}\t{5:1.10f}\t{6:1.10f}\t{7:1.10f}\t{8:1.10f}\t{9:1.10f}'.format(category, *stats))

            f.write('\n\nOverall :\t{0:1.10f}\t{1:1.10f}\t{2:1.10f}\t{3:1.10f}\t{4:1.10f}\t{5:1.10f}\t{6:1.10f}\t{7:1.10f}\t{8:1.10f}'.format(*[mean(z) for z in zip(*overallStats)]))
            if len(categoryStats) < 6:
                f.write('\nYour method will not be visible in the Overall section.\nYou need all 6 categories to appear in the overall section.')
