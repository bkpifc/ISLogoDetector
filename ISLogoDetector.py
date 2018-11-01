"""
* !/usr/bin/env python2
* IS Logo Detector
* 24.10.2018
* Lukas Burkhardt
* Adapted from Tensorflow Object Detection Sample Script
"""


import numpy as np
import os
import sys
import tensorflow as tf
import hashlib
import ConfigParser
import cv2
import magic
from distutils.version import StrictVersion
from PIL import Image
from datetime import datetime
from multiprocessing import Pool

startTime = datetime.now()


######
#
# Model and Variable Preparation
#
######

# Variable to determine minimum GPU Processer requirement & to disable TF log output
#os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '5'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Validating TF version
if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

# Initiating Config Parsing
configParser = ConfigParser.RawConfigParser()
configFilePath = r'config.txt'
configParser.read(configFilePath)

# Defining multiple needed variables based on config file paths & adding object_detection directory to path
MODELPATH = configParser.get('regular-config','MODELPATH')
PATH_TO_TEST_IMAGES_DIR = configParser.get('regular-config','PATH_TO_TEST_IMAGES_DIR')
PATH_TO_RESULTS = configParser.get('regular-config','PATH_TO_RESULTS')
PATH_TO_OBJECT_DETECTION_DIR = configParser.get('regular-config', 'PATH_TO_OBJECT_DETECTION_DIR')
PATH_TO_FROZEN_GRAPH = MODELPATH + '/frozen_inference_graph.pb'
IMAGENAMES = os.listdir(PATH_TO_TEST_IMAGES_DIR)
TEST_IMAGE_PATHS = [PATH_TO_TEST_IMAGES_DIR + '/' + i for i in IMAGENAMES]
sys.path.append(PATH_TO_OBJECT_DETECTION_DIR)
frames_per_video = 10 #frames to analyze per video file

# Loading the frozen model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


######
#
# Worker function to prepare and reshape the input videos to a Numpy array
# and to calculate the MD5 hashes of them.
# The function analyzes as much frames as indicated in the variable "frames_per_video" (Default = 10)
#
######

def load_video_into_numpy_array(image_path):

    videoframes = []
    # Loading the video via the OpenCV framework
    try:
        vidcap = cv2.VideoCapture(image_path)
        im_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        im_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Hashing the image once
        hash_md5 = hashlib.md5()
        with open(image_path, "rb") as f:
             for chunk in iter(lambda: f.read(4096), b""):
                 hash_md5.update(chunk)
        hashvalue = hash_md5.hexdigest()

        # Extracting the frames from the video
        for percentile in range(0, frames_per_video):
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, float(float(percentile) / frames_per_video))
            success, extracted_frame = vidcap.read()
            extracted_frame = cv2.cvtColor(extracted_frame, cv2.COLOR_BGR2RGB)
            # And reshape them into a numpy array
            np_array = np.array(extracted_frame).reshape(
                (im_height, im_width, 3)).astype(np.uint8)

            cluster = hashvalue, np_array
            videoframes.append(cluster)

        vidcap.release()
        return videoframes

    except cv2.error:
        print "Could not process video: " + str(image_path)




######
#
# Worker function to prepare and reshape the input images into a Numpy array
# and to calculate the MD5 hashes of them.
#
######

def load_image_into_numpy_array(image_path):


  try:
    # Open, measure and convert image to RGB channels
    image = Image.open(image_path)
    (im_width, im_height) = image.size

    image = image.convert('RGB')
    np_array = np.array(image.getdata()).reshape(
                (im_height, im_width, 3)).astype(np.uint8)

    # Hash the image in byte-chunks of 4096
    hash_md5 = hashlib.md5()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    hashvalue = hash_md5.hexdigest()

    return hashvalue, np_array


  # Throw errors to stdout
  except IOError:
    # If image file cannot be read, check if it is a video
    if str(magic.from_file(image_path, mime=True))[:5] == 'video':
        # If so, return a video flag instead of numpy array
        flag = "VIDEO"
        return image_path, flag

    else:
        print "Could not open image: " + str(image_path)

  except:
    print "General error with file: " + str(image_path)


######
#
# Detection within loaded images
# Creation of file with hashes and detection scores
#
######

def run_inference_for_multiple_images(images, graph, hashvalues):

    # Initiate variables
    detectedLogos = 0
    processedImages = 0
    errorcount = 0

    # Create TF Session with loaded graph
    with graph.as_default():
        with tf.Session() as sess:
            #Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                 'num_detections', 'detection_scores',
            ]:
               tensor_name = key + ':0'
               if tensor_name in all_tensor_names:
                 tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                     tensor_name)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Prepare results file with headers
            output_dicts = []
            detectionr = open(PATH_TO_RESULTS + "/Detection_Results.csv", 'w')
            detectionr.write('hash,score\n')

            # Setting the detection limit to 90% - lower values will be discarded
            detectionlimit = 0.9

            # Conduct actual detection within single image
            for index, image in enumerate(images):
                try:
                    #print image
                    processedImages += 1
                    output_dict = sess.run(tensor_dict,
                                           feed_dict={image_tensor: np.expand_dims(image, 0)})

                    # all outputs are float32 numpy arrays, so convert types as appropriate
                    output_dict['num_detections'] = int(output_dict['num_detections'][0])
                    output_dict['detection_scores'] = output_dict['detection_scores'][0]
                    detectionhit = output_dict['num_detections']

                    hashvalue = hashvalues[index]

                    # Validate against detection limit (default: 90%) and write hash/score if above
                    for j in range(detectionhit):
                        score = output_dict['detection_scores'][j]
                        if (score >= detectionlimit):
                            scorestring = str(score)
                            detectedLogos += 1
                            line = ",".join([hashvalue, scorestring])
                            detectionr.write(line + "\n")

                except tf.errors.InvalidArgumentError:
                    errorcount += 1
                    print "Unable to process file dimensions of file with hash: " + str(hashvalue)

            return processedImages, detectedLogos, errorcount


            detectionr.flush()
            detectionr.close()


######
#
# Main program function which first loads images and then starts detection
#
######

# Print starting time to stdout
print str(datetime.now()) + ": Process started. Loading images..."


if __name__ == '__main__':

    # Initiate needed variables
    vidlist = []
    final_images = []

    # Multiprocess the image load function on all CPU cores available
    pool = Pool()
    processed_images = pool.map(load_image_into_numpy_array, TEST_IMAGE_PATHS)
    pool.close()
    # Synchronize after completion
    pool.join()

    # Clean the result for None types (where image conversion failed)
    processed_images = [x for x in processed_images if x != None]

    # Check for the video flag
    for processed_image in processed_images:
        if str(processed_image[1]) == "VIDEO":
            # If present, populate the video list
            vidlist.append(processed_image[0])

        else:
            # If not, put it to the final images list
            final_images.append(processed_image)

    # Multiprocess the video load function on all CPU cores available
    pool = Pool()
    videoframes = pool.map(load_video_into_numpy_array, vidlist)
    pool.close()
    # Synchronize after completion
    pool.join()

    # Clean the result for None types (where video conversion failed)
    for video in videoframes:
        if video is not None:
            final_images.extend(video)


    # Split the result from the loading function into hashes and image arrays
    hashvalues, image_nps = zip(*final_images)


# Print starting time of detection to stdout
print str(datetime.now()) + ": Loading completed. Detecting..."

# Execute detection
processedImages, detectedLogos, errorcount = run_inference_for_multiple_images(image_nps, detection_graph, hashvalues)

# Print process statistics to stdout
print "Results: " + configParser.get('regular-config', 'PATH_TO_RESULTS') + "/Detection_Results.csv"
print "Total Amount of Files: " + str(len(TEST_IMAGE_PATHS))
print "Processed Images: " + str(processedImages - (len(vidlist) * frames_per_video))
print "Processed Videos: " + str(len(vidlist))
print "Detected potential IS Logos: " + str(detectedLogos)
print "Error during detection: " + str(errorcount)
print "Processing time: " + str(datetime.now() - startTime)





