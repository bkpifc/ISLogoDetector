# ISLogoDetector
This trained tensorflow model allows you to detect logos. It does so in a folder full of images. The input consists of a folder which contains all sorts of images (jpg, png and certain bmp/gifs supported) and generates a CSV file with the hash-values (MD5) and the score of the images where the system detected one or more logos. However, with this model you can also enable real-time detection via webcams, for instance.

## Data structure:
- Script (ISLogoDetector.py)
- Config File (config.txt)
- Model (complete frozen TF graph)


## Dependencies:
- Installed Tensorflow with ObjectDetection API: [Installation Instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
- Python 2.7 (including Numpy, ConfigParser, MatplotLib, PIL)

## Usage:
0) Download and extract the [latest release](https://github.com/bkpifc/ISLogoDetector/releases)

1) Make sure your Tensorflow Research directory (tensorflow/models/research) is added to pythonpath:

`export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim`

2) Export the relevant images to a directory

3) Configure the settings within config.txt:
   - MODELPATH → Path to the ISLogoDetectorModel directory (see releases)
   - PATH_TO_OBJECT_DETECTION_DIR → Folder in which object_detection resides (usually tensorflow/models/research)
   - PATH_TO_TEST_IMAGES_DIR → Folder in which the images are stored
   - PATH_TO_RESULTS → Where the output csv shall be stored

4) Execute ISLogoDetector.py in the same folder as the config.txt file:

`./ISLogoDetector.py`

For guidance on re-training the model, or to create a fully new tensorflow model, please see those various excellent postings or contact us directly: 

[Step by step tensorflow object detection api tutorial](https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e)

[TF Object Detection Model Training](https://gist.github.com/douglasrizzo/c70e186678f126f1b9005ca83d8bd2ce)
