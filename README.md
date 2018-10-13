# ISLogoDetector
Tensorflow Project to detect specific Logos

Dependencies:
- Installed Tensorflow with ObjectDetection API 
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
- Python 2.7 (including Numpy, ConfigParser, MatplotLib, PIL)


Content:
ISLogoDetectorModel3 - Directory containing the trained, frozen TF model
- frozen_inference_graph.bp = Frozen Model
- is_label_map.pbtxt = List of labels to be assigned (only 1 in this case)
- model.ckpt = Checkpoint

ISLogoDetectorScript - Directory containing the script and the config file
- config.txt = Textfile to configure paths of various required locations (images, model, output)
- ISLogoDetector.py = Python file to be executed (without any flags)


Usage:
Place your image files into a folder (flat structure) and configure the config.txt file accordingly.
Run the python script and wait for the resulting "Detection_Results.csv" to be dropped at the location you specified.
Output are csv values (hash,score).


In order to train your own (or this model further) please see
xxxx

