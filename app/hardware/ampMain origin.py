import cv2
import os
import numpy as np
import random
import statistics
import time
import math
#from acousticbot2 import acousticBot2
from exif import Image as exifImage
from app.hardware.munkres_solver import MunkresSolver
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#helper
def imageLoop(image):
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    '''注释
	参数1：窗口的名字
	参数2：窗口类型，CV_WINDOW_AUTOSIZE 时表明窗口大小等于图片大小。不可以被拖动改变大小。
	CV_WINDOW_NORMAL 时，表明窗口可以被随意拖动改变大小。
	'''

    while True:
        cv2.imshow("frame", image)
        query = cv2.waitKey(1)
        if query == ord('q'):#returns an integer representing the Unicode character.

            break
        
    cv2.destroyAllWindows()

#helper
def saveImageWithMetaData(filename, image, comment):   
    """Saves an image and adds comment as metadata"""
    # write image under filename
    cv2.imwrite(filename, image)
    # wait, to be sure image has been written
    time.sleep(0.01)

    # open image file that was just written
    with open(filename, 'rb') as image_file:
        # open image with exif 
        my_image = exifImage(image_file)
        # Add comment containing experiment info to metadata
        my_image.image_description = comment
        
    # Write image with metadata over previous write
    with open(filename, 'wb') as new_image_file:
        new_image_file.write(my_image.get_file())
        
    return 1

##helper
def getTempDataPath():
    mainpath = os.getcwd()
    
    return mainpath

## TODO
def loadData():
    pass

#helper
def initBlobDetector():
    params = cv2.SimpleBlobDetector_Params()
    
    params.minThreshold = 10
    params.maxThreshold = 1000
    # Color: 0 for dark, 225 light
    params.filterByColor = False
    params.blobColor = 200
    params.filterByArea = True
    params.minArea = 150
    params.maxArea = 1500
    params.filterByCircularity = False
    params.minCircularity = 0.1
    params.filterByConvexity = False
    params.minConvexity = 0.87
    params.filterByInertia = False
    params.minInertiaRatio = 0.01
    
    detector = cv2.SimpleBlobDetector_create(params)
    
    return detector


#helper
def createBlobDetector(params=None):
    if params:
        detector = cv2.SimpleBlobDetector_create(params)
    else:
        detector = initBlobDetector()
        
    return detector

#helper
def getParticleLocations(image, detector):
    keypoints = detector.detect(image)

    # Create array of coordinates of detected particles from keypoints
    coordinates = []
    for keypoint in keypoints:
        coords = (keypoint.pt[0], keypoint.pt[1])
        coordinates.append(coords)

    cv2.waitKey(1)
    
    return keypoints, coordinates

# helper function for choosing new amplitude
def chooseNextAmp(prevMovements, prevAmps, desired_stepSize, default_amp, max_increase, min_amp, max_amp):
    # median of element-wise divison
    k = statistics.median(prevAmps / prevMovements)        
    # tune k by given step
    ret = k * desired_stepSize                
    # choose largest amp that yields movement smaller than step
    try:
        last_good_amp = max(prevAmps[(prevMovements < desired_stepSize)])
    except ValueError: 
        # resort to default if empty 
        last_good_amp = default_amp    
        
    # choose between ret or an increased amp from the best tried amp
    newAmp = min(ret, last_good_amp * max_increase)
    
    # check newAmp is within boundaries for amp and return
    return max(min(newAmp, max_amp),min_amp)

## conducts one experiment
def doAmpExperiment(acoBot, beforejpg,afterjpg,freq,amp,duration,desired_particles,simulate=0):
        """Runs experiment, which includes: 
            1. Taking picture of current state (and enforcing particle amount) 
            2. playing a frequency,
            3. taking picture new state
        """    
        parameters = str({'Freq': freq, 'Amp': amp, 'Duration': duration})
        #comment = '{}, {}, {}'.format(freq,amp,duration)
        detector = createBlobDetector()
        
        while(True):
            img = acoBot.getImage(croppingCoordinates=[(400,70), (1450,1100)])
            if not desired_particles:
                break
            _, coords = getParticleLocations(img, detector)
            num_particles = len(coords)
            if (num_particles == desired_particles):
                break
            
            tempImg = img.copy()
            for coord in coords:
                tempImg = cv2.circle(img, (int(coord[0]), int(coord[1])), 5, (0,0,0), 2)
            imageLoop(tempImg)
            
            input('Too few particles on the plate, please add more! Hit enter when ready. Is: {} Should be: {}\n'.format(num_particles,desired_particles))
        
        #cv2.imwrite(img, beforejpg,'jpg','Comment',comment)
        #cv2.imwrite(beforejpg, img)
        saveImageWithMetaData(beforejpg, img, parameters)
        
        acoBot.playSignal(freq, amp, duration/1000)
        time.sleep(duration / 1000 + 0.5) #For under water + 3, For air + 0.5       
        
        img2 = acoBot.getImage(croppingCoordinates=[(400,70), (1450,1100)])
        #cv2.imwrite(img2, afterjpg, 'jpg', 'Comment', comment)
        #cv2.imwrite(afterjpg, img2)
        saveImageWithMetaData(afterjpg, img2, parameters)
        
        return num_particles
    
## Loads before and after images, computes assignments, and calculates movement
def loadAmpExp(beforeFile, afterFile):
    # Create detector for particle detection
    detector = createBlobDetector()
    # Get positions of particles before and after 
    _, p_before = getParticleLocations(cv2.imread(beforeFile), detector)
    _, p_after = getParticleLocations(cv2.imread(afterFile), detector)
    
    # Match particle positions before with positions after using munkres algorithm
    m = MunkresSolver()
    # assignments in form: list:((x1,y1), euclidianDist, (x2,y2))
    assignments = m.getAssignments(p_before, p_after)
    dif = [x[1] for x in assignments]
    
    # get movement from data
    absdif = math.sqrt(sum([x**2 for x in dif]))
    #movement = quantile(absdif,0.75)
    movement = m.quantile(absdif, 0.75)
    
    # Open before-image to fetch metadata on experiment
    with open(beforeFile, 'rb') as image_file:
        my_image = exifImage(image_file)   
        # eval string-comment to dictionary with metadata
        metadata = eval(my_image.image_description)
        
    # Fetch experiment parameters from metadata
    freq = metadata['Freq']
    amp = metadata['Amp']
    duration = metadata['Duration']
    
    #[movement, freq, amp, duration, p_before, p_after]
    return [movement, freq, amp, duration]

## TODO
def plotAmpData(datafile):
    pass

## TODO
def makeModeInfo(datafile):
    pass

def ampMain(acoBot,parameters):
    print('starting')
    simulate = parameters['simulate']
    id = parameters['id']
    desired_particles = parameters['desired_particles']
    desired_stepSize = parameters['desired_stepSize']
    cycles = parameters['cycles']
    minfreq = parameters['minfreq']
    maxfreq = parameters['maxfreq']
    duration = parameters['duration']
    default_amp = parameters['default_amp']
    min_amp = parameters['min_amp']
    max_amp = parameters['max_amp']
    max_increase = parameters['max_increase']
    exps_before_reset = parameters['exps_before_reset']
    basescale = parameters['basescale']
    print(id)
    # simulate = 0 # 1 = generated random images, 0 = get images over http
    # id = 'acoBot2_amptest10' # Identifier of the experiment run
    # desired_particles = 2 # How many particles should be on the plate, at least, for the experiment to start
    # desired_stepSize = 10 # The experiment tries to adjust the amplitudes so that the 75% of the particles move less than this
    # cycles = 2 # For each frequency, how many PTV steps is taken in total. The total number of exps cycles * number of frequencies
    # minfreq = 500 # All notes from the scale below this frequency are discarded.
    # maxfreq = 1000 # All notes from the scale above this frequency are discarded
    # duration = 500 # in milliseconds, constant for all notes
    # default_amp = 0.1 # starting amplitude: for 2*3*5 actuators = 0.02 for 2*3*20 actuators = 0.005
    # min_amp = 0.01 # never decrease amplitude below this
    # max_amp = 2 # never increase amplitude above this
    # max_increase = 1.5 # never increase the amplitude more than 1.5 x from the previous experiment
    # exps_before_reset = 10 # The balls are replaced to good locations every this many cycles
    # basescale = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88] # C major
    # basescale = [261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88] # chromatic 

    # create prefix for all images saved from experiments
    prefiximg = getTempDataPath() + '\\' +'ampExp'+'\\'+ id + '\\'
    # Create data directory for experiments
    os.makedirs(prefiximg)
    os.chdir(prefiximg) #change directory to prefiximg
    
    # Create data file string
    datafile = prefiximg + id + '.mat' 
    
    # create octaves below and above frequencies in scale  
    tmpscale = [(x * 2**y) for y in range(-10, 11) for x in basescale]   
    # Filter out frequencies above/below maxfreq/minfreq
    expfreq = np.array([freq for freq in tmpscale if maxfreq > freq and freq > minfreq])
        
    # number of frequencies (52)
    M = len(expfreq)        

    # exps to play every frequency N (30) times [30*52]
    exps = []
    # loop over cycles and add permutation of freqs, results in M*cycles exps
    for i in range(cycles):
        # create permutations of frequency sequences
        exps = exps + random.sample(range(M), k=M)
    exps = np.array(exps).astype(int)
    
    # counter for keeping track of experiments completed
    exp_counter = 0
    # amount of expriments
    N = len(exps)
    # tolerance for ?
    tol = 1e-6

    # get data file if there is one
    if (os.path.exists(datafile)):
        loadData(datafile)
    else:   
        # create empty arrays for experiments
        movements = np.full(N, np.nan)
        freqs = np.full(N, np.nan)
        amps = np.full(N, np.nan)
        durations = np.full(N, np.nan)
        
    # loop over experiments (N = 30*52)
    for i in range(N):
        # if amps is not full 
        if np.isnan(amps[i]):
            # create image filenames for before and after
            beforefile = '{:s}{:s}_{:d}_{:.0f}_{:.0f}_in.jpg'.format(prefiximg,id,i+1,expfreq[exps[i]],duration)
            afterfile = '{:s}{:s}_{:d}_{:.0f}_{:.0f}_out.jpg'.format(prefiximg,id,i+1,expfreq[exps[i]],duration)
            # if the files don't already exist, create them by running exps
            if not (os.path.exists(beforefile) and os.path.exists(afterfile)):
                # get wanted frequency for experiment
                freq = expfreq[exps[i]]
                # Boolean mask for succesful experiments (|?)
                #选择频率为freq的实验
                ind = ~np.isnan(amps) & (abs(freq - freqs) < tol) & (abs(durations - duration) < tol)
                # get preiously used amplitudes for experiments
                prevAmps = amps[ind]
                # if none take default
                if prevAmps.size == 0:
                    amp = default_amp
                else:
                    # get movement from prev exp
                    prevMovements = movements[ind]
                    # scale amp depending on movement of prev exp
                    amp = chooseNextAmp(prevMovements, prevAmps, desired_stepSize, default_amp, max_increase, min_amp, max_amp)
                    
                # give user possibility to reset particles
                if (exp_counter >= exps_before_reset):
                    # prompt user to redistribute particles
                    input('Redistribute the particles evenly on the plate and hit enter when ready\n')
                    exp_counter = 0
                    
                # take picture for experiment + misc.
                doAmpExperiment(acoBot, beforefile,afterfile,freq,amp,duration,desired_particles,simulate)
                exp_counter = exp_counter + 1

            # get data from pictures for experiment
            [movements[i],freqs[i],amps[i],durations[i]] = loadAmpExp(beforefile,afterfile)
            # save data to datafile
            #save(datafile, 'expfreq', 'exps', 'movements', 'freqs', 'amps', 'durations')              
            # print progress
            print("{:d} / {:d} ({:.0f}% done) {:1.1f} {:1.3f} {:1.4f}\n".format(i+1, N, (i+1)*100/N, freqs[i], amps[i], movements[i]))   
        
        # plot the data for every 100 experiments
        #if (i % 100 == 0):
            # plots amplitude/freq, amplitude/exp, movement/freq, ...
            #plotAmpData(datafile)   
    
    # Plot movement amplitude convergence over movement from experiments
    expsData = {'expNum': range(N),'Frequencies': freqs, 'Amplitudes': amps, 
                'Movements': movements, 'Durations': durations}
    expsData = pd.DataFrame.from_dict(expsData)
    # plot with seaborn and use the hue parameter
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='expNum', y='Amplitudes', data=expsData, hue='Frequencies', legend='full')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.xticks(rotation=90)
    plt.show()
    
    
    # Save experiment data to mat-file
    expsData = {'Frequencies': freqs, 'Amplitudes': amps, 
                'Movements': movements, 'Durations': durations}
    expsData = pd.DataFrame.from_dict(expsData)
    expsData.to_csv('expsData.csv')
    
    # Save freq/amplitudes to mat-file
    finalAmps = [binnedAmps[-1] for binnedAmps in [amps[[i for i in range(len(amps)) if freqs[i] == x]] for x in expfreq]]
    freqAmpData = {'frequencies': expfreq, 'Amplitudes': finalAmps}
    freqAmpData = pd.DataFrame.from_dict(freqAmpData)
    freqAmpData.to_csv('freqAmpData.csv')
    
    # Create file with all data from exps
    #makeModeInfo(datafile)
    

# global acoBot 
# acoBot = acousticBot2()
# acoBot.enableCamera()
# acoBot.startCapture()

if __name__ == '__main__':
    ampMain()
