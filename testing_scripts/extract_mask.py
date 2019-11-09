
"""
Created on Mon Jun 27 2018

@author: longang
"""

# coding: utf-8
#get_ipython().magic(u'load_ext autotime')
import numpy as np
import os, glob, sys
from keras.preprocessing import image as kImage
#from skimage.transform import pyramid_gaussian
from keras.models import load_model
from scipy.misc import imsave#, imresize
import gc


# Optimize to avoid memory exploding. 
# For each video sequence, we pick only 1000frames where res > 400
# You may modify according to your memory/cpu spec.
def checkFrame(X_list):
    img = kImage.load_img(X_list[0])
    img = kImage.img_to_array(img).shape # (480,720,3)
    num_frames = len(X_list) # 7000
    max_frames = 1000 # max frames to slice
    if(img[1]>=400 and len(X_list)>max_frames):
        print ('\t- Total Frames:' + str(num_frames))
        num_chunks = num_frames/max_frames
        num_chunks = int(np.ceil(num_chunks)) # 2.5 => 3 chunks
        start = 0
        end = max_frames
        m = [0]* num_chunks
        for i in range(num_chunks): # 5
            m[i] = range(start, end) # m[0,1500], m[1500, 3000], m[3000, 4500]
            start = end # 1500, 3000, 4500 
            if (num_frames - start > max_frames): # 1500, 500, 0
                end = start + max_frames # 3000
            else:
                end = start + (num_frames- start) # 2000 + 500, 2500+0
        print ('\t- Slice to:' + str(m))
        del img, X_list
        return [True, m]
    del img, X_list
    return [False, None]
    
# Load some frames (e.g. 1000) for segmentation
def generateData(scene_input_path, X_list, scene):
    # read images
    X = []
    print ('\n\t- Loading frames:')
    for i in range(0, len(X_list)):
        img = kImage.load_img(X_list[i])
        x = kImage.img_to_array(img)
        X.append(x)
        sys.stdout.write('\b' * len(str(i)))
        sys.stdout.write('\r')
        sys.stdout.write(str(i+1))
    
    del img, x, X_list
    X = np.asarray(X)
    print ('\nShape' + str(X.shape))
    
# For FgSegNet (multi-scale)
#    s2 = []
#    s3 = []
#    num_img = X.shape[0]
#    prev = 0
#    print ('\t- Downscale frames:')
#    for i in range(0, num_img):
#       pyramid = tuple(pyramid_gaussian(X[i]/255., max_layer=2, downscale=2))
#       s2.append(pyramid[1]*255.)
#       s3.append(pyramid[2]*255.)
#       sys.stdout.write('\b' * prev)
#       sys.stdout.write('\r')
#       s = str(i+1)
#       sys.stdout.write(s)
#       prev = len(s)
#       del pyramid
#    s2 = np.asarray(s2)
#    s3 = np.asarray(s3)
#    print ('\n')
#    print (s1.shape, s2.shape, s3.shape)

#    return [X, s2, s3] #return for FgSegNet (multi-scale)

    return X #return for FgSegNet_v2

def getFiles(scene_input_path):
    inlist = glob.glob(os.path.join(scene_input_path,'*.jpg'))
    return np.asarray(inlist)



# Extract all mask
dataset = {
           'baseline':[
                   'highway', 
                   'pedestrians',
                   'office',
                   'PETS2006'
                   ],
           'cameraJitter':[
                   'badminton',
                   'traffic',
                   'boulevard', 
                   'sidewalk'
                   ],
           'badWeather':[
                   'skating', 
                   'blizzard',
                   'snowFall',
                   'wetSnow'
                   ],
            'dynamicBackground':[
                   'boats',
                   'canoe',
                   'fall',
                   'fountain01',
                   'fountain02',
                    'overpass'
                    ],
            'intermittentObjectMotion':[
                    'abandonedBox',
                    'parking',
                    'sofa',
                    'streetLight',
                    'tramstop',
                    'winterDriveway'
                    ],
            'lowFramerate':[
                    'port_0_17fps',
                    'tramCrossroad_1fps',
                    'tunnelExit_0_35fps',
                    'turnpike_0_5fps'
                    ],
            'nightVideos':[
                    'bridgeEntry',
                    'busyBoulvard',
                    'fluidHighway',
                    'streetCornerAtNight',
                    'tramStation',
                    'winterStreet'
                    ],
           'PTZ':[
                   'continuousPan',
                   'intermittentPan',
                   'twoPositionPTZCam',
                   'zoomInZoomOut'
                   ],
            'shadow':[
                    'backdoor',
                    'bungalows',
                    'busStation',
                    'copyMachine',
                    'cubicle',
                    'peopleInShade'
                    ],
           'thermal':[
                   'corridor',
                   'diningRoom',
                   'lakeSide',
                   'library',
                   'park'
                   ],
            'turbulence':[
                    'turbulence0',
                    'turbulence1',
                   'turbulence2',
                   'turbulence3'
                    ] 
}

# number of exp frame (25, 50, 200)
num_frames = 25

# 1. Raw RGB frame to extract foreground masks, downloaded from changedetection.net
raw_dataset_dir = 'CDnet2014_dataset'

# 2. model dir
main_mdl_dir = os.path.join('FgSegNet_v2', 'models' + str(num_frames))

# 3. path to store results
results_dir = os.path.join('FgSegNet_v2', 'results' + str(num_frames))


# Loop through all categories (e.g. baseline)
for category, scene_list in dataset.items():
    # Loop through all scenes (e.g. highway, ...)
    for scene in scene_list:
        print ('\n->>> ' + category + ' / ' + scene)
        mdl_path = os.path.join(main_mdl_dir, category , 'mdl_' + scene + '.h5')
        
        mask_dir = os.path.join(results_dir, category, scene)
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        
        # path of dataset downloaded from CDNet
        scene_input_path = os.path.join(raw_dataset_dir, category, scene, 'input')
        # path of ROI to exclude non-ROI
        # make sure that each scene contains ROI.bmp and have the same dimension as raw RGB frames
        ROI_file = os.path.join(raw_dataset_dir, category, scene, 'ROI.bmp')
        
        # refer to http://jacarini.dinf.usherbrooke.ca/datasetOverview/
        img = kImage.load_img(ROI_file, grayscale=True)
        img = kImage.img_to_array(img)
        img = img.reshape(-1) # to 1D
        idx = np.where(img == 0.)[0] # get the non-ROI, black area
        del img
        
        # load path of files
        X_list = getFiles(scene_input_path)
        if (X_list is None):
            raise ValueError('X_list is None')

        # slice frames
        results = checkFrame(X_list)
        
        # load model to segment
        model = load_model(mdl_path)

        # if large numbers of frames, slice it
        if(results[0]): 
            for rangeee in results[1]: # for each slice
                slice_X_list =  X_list[rangeee]

                # load frames for each slice
                data = generateData(scene_input_path, slice_X_list, scene)
                
                # For FgSegNet (multi-scale only) 
                #Y_proba = model.predict([data[0], data[1], data[2]], batch_size=batch_size, verbose=1) # (xxx, 240, 320, 1)
                
                # For FgSegNet_v2
                Y_proba = model.predict(data, batch_size=1, verbose=1)
                del data

                # filter out
                shape = Y_proba.shape
                Y_proba = Y_proba.reshape([shape[0],-1])
                if (len(idx)>0): # if have non-ROI
                    for i in range(len(Y_proba)): # for each frames
                        Y_proba[i][idx] = 0. # set non-ROI pixel to black
                        
                Y_proba = Y_proba.reshape([shape[0], shape[1], shape[2]])

                prev = 0
                print ('\n- Saving frames:')
                for i in range(shape[0]):
                    fname = os.path.basename(slice_X_list[i]).replace('in','bin').replace('jpg','png')
                    x = Y_proba[i]
                    
#                    if batch_size in [2,4] and scene=='badminton':
#                        x = imresize(x, (480,720), interp='nearest')
#                        
#                    if batch_size in [2,4] and scene=='PETS2006':
#                        x = imresize(x, (576,720), interp='nearest')
                    
                    imsave(os.path.join(mask_dir, fname), x)
                    sys.stdout.write('\b' * prev)
                    sys.stdout.write('\r')
                    s = str(i+1)
                    sys.stdout.write(s)
                    prev = len(s)
                    
                del Y_proba, slice_X_list

        else: # otherwise, no need to slice
            data = generateData(scene_input_path, X_list, scene)
            
            # For FgSegNet (multi-scale)
            #Y_proba = model.predict([data[0], data[1], data[2]], batch_size=batch_size, verbose=1) # (xxx, 240, 320, 1)
            
            # For FgSegNet_v2
            Y_proba = model.predict(data, batch_size=1, verbose=1)
            
            del data
            shape = Y_proba.shape
            Y_proba = Y_proba.reshape([shape[0],-1])
            if (len(idx)>0): # if have non-ROI
                    for i in range(len(Y_proba)): # for each frames
                        Y_proba[i][idx] = 0. # set non-ROI pixel to black

            Y_proba = Y_proba.reshape([shape[0], shape[1], shape[2]])
            
            prev = 0
            print ('\n- Saving frames:')
            for i in range(shape[0]):
                fname = os.path.basename(X_list[i]).replace('in','bin').replace('jpg','png')
                x = Y_proba[i]
                
#                if batch_size in [2,4] and scene=='badminton':
#                    x = imresize(x, (480,720), interp='nearest')
#                
#                if batch_size in [2,4] and scene=='PETS2006':
#                        x = imresize(x, (576,720), interp='nearest')
                        
                imsave(os.path.join(mask_dir, fname), x)
                sys.stdout.write('\b' * prev)
                sys.stdout.write('\r')
                s = str(i+1)
                sys.stdout.write(s)
                prev = len(s)
            del Y_proba
        del model, X_list, results

    gc.collect()