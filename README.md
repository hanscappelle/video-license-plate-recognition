# Video License Plate Recognition with

A python script that takes a video mp4 file as input and checks for license plates frame by frame
resulting in video output with text based info overlay and a CSV text file. 

<img src="https://github.com/hanscappelle/video-license-plate-recognition/blob/main/Screenshot%202025-06-17%20at%2023.43.15.png"/>

## Install dependencies

Instructions for using on a Mac with Python3 installed by homebrew. 
Note my system already had Python 3.13 installed which is NOK for tensorflow.
See tensorflow installation info at https://www.tensorflow.org/install

```
(venv) hcpl@Mac Video-LPR % python3 --version
Python 3.13.5

(venv) hcpl@Mac Video-LPR % python3 -m pip --version
pip 25.1.1 from /video-license-plate-recognition/path/to/venv/lib/python3.13/site-packages/pip (python 3.13)
```

I had to downgrade my python version to something matching tensorflow requirements of `3.9 - 3.11`

```
brew install python@3.11
```

Then reference that specific version with these commands instead: `python3.11` and `pip3.11`. 
At that point installation of tensorflow and other dependencies was possible with:

```
pip3.11 install tensorflow
pip3.11 install opencv-python
pip3.11 install ultralytics
pip3.11 install easyocr
```

## Execute this code

Find out what arguments are supported:

```
python3.11 main.py -h                               
usage: main.py [-h] [sourcePath] [outputPath] [skipFrames] [resHorizontal] [resVertical] [outputCsvPath] [confidenceLimit]

License Plate Scanner

positional arguments:
  sourcePath
  outputPath
  skipFrames
  resHorizontal
  resVertical
  outputCsvPath
  confidenceLimit


options:
  -h, --help     show this help message and exit
```

Example for no frame skipping working with a downsampled video in 1280x960 resolution 
and output text only if confidence is above 0.3.

```
python3.11 main.py input.mp4 output.mp4 1 1280 960 output.csv 0.3
```

Example output running above command

```
(venv) hcpl@Mac Video-LPR % python3 main.py         
Using CPU. Note: This module is much faster with a GPU.
Downloading detection model, please wait. This may take several minutes depending upon your network connection.
Progress: |██████████████████████████████████████████████████| 100.0% CompleteDownloading recognition model, please wait. This may take several minutes depending upon your network connection.
Progress: |██████████████████████████████████████████████████| 100.0% Complete% (venv) hcpl@Mac Video-LPR % 
```

## Improvements

* don't wait till the end to export csv text based results
* create a better model? Or test for better OCR options
* it probably doesn't help that my source video was anamorphic and desqueezed

### Done

* add parameters for more options
* also output text based in CSV format
* limiting to allowed characters with
  allowlist (string) - Force EasyOCR to recognize only subset of characters. Useful for specific problem (E.g. license plate, etc.)
  from https://www.jaided.ai/easyocr/documentation/
  0123456789-ABCDEFGHIJKLMNOPQRSTUVWXYZ
* also write detected frames https://roboflow.com/use-opencv/save-an-image-with-imwrite

## References

Source for original version of this code found in this article (see _main.py in this project for reference): 
https://medium.com/@mahijain9211/license-plate-detection-from-video-files-using-yolo-and-easyocr-6b647f0c94d5

Source for model referenced in original article (don't use that model): 
https://huggingface.co/Snearec/detectorMalezasYolo8/blob/2332b15b097b3f9f94fc5a260d59dae1e1b8c443/best_float32.tflite

Source for alternative model that does a way better job: 
https://github.com/sveyek/Video-ANPR

Adapted script to receive command line arguments using this info:
https://www.tutorialspoint.com/python/python_command_line_arguments.htm

And added some extra output besides the video.

## Alternative implementations

Just some similar projects I found. 

https://www.geeksforgeeks.org/python/detect-and-recognize-car-license-plate-from-a-video-in-real-time/

https://github.com/wavelolz/Video-License-Plate-Recognition

https://github.com/sveyek/Video-ANPR

https://github.com/mendez-luisjose/License-Plate-Detection-with-YoloV8-and-EasyOCR

