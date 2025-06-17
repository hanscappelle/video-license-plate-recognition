# Video License Plate Recognition


## install dependencies

on mac using Python3 installed with home-brew. Note my system already had Python 3.13 installed which is too high for tensorflow

```
(venv) hcpl@Mac Video-LPR % python3 --version
Python 3.13.5

(venv) hcpl@Mac Video-LPR % python3 -m pip --version
pip 25.1.1 from /Users/hcpl/Development/Video-License-Plate-Recognition-main/path/to/venv/lib/python3.13/site-packages/pip (python 3.13)
```

See tensorflow installation info at https://www.tensorflow.org/install

I had to downgrade my python version to something matching tensorflow requirements of 3.9-3.11

```
brew install python@3.11
```

then reference that specific version with

python3.11 and pip3.11 commands

so at that point installation of tensorflow was possible with

```
pip3.11 install tensorflow
pip3.11 install opencv-python
pip3.11 install ultralytics
pip3.11 install easyocr
```

## execute code

Find out what arguments are supported

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

Example for no frame skipping with 1280x960 resolution (resampled to) and output text if confidence above 0.3


```
python3.11 main.py input.mp4 output2.mp4 1 1280 960 output.csv 0.3
```

When using with no args you'll have to make sure path is found and so on

```
python3 main.py
```

after changing path of video to process (default is set to input.mp4)

generates output_video.mp4

example output

```
(venv) hcpl@Mac Video-LPR % python3 main.py         
Using CPU. Note: This module is much faster with a GPU.
Downloading detection model, please wait. This may take several minutes depending upon your network connection.
Progress: |██████████████████████████████████████████████████| 100.0% CompleteDownloading recognition model, please wait. This may take several minutes depending upon your network connection.
Progress: |██████████████████████████████████████████████████| 100.0% Complete% (venv) hcpl@Mac Video-LPR % 
```

## Improvements

* don't wait till the end to export csv
* create a better model? Or test for better OCR

## References

Source for original version of this code found in this article (see _main.py in this project for reference): https://medium.com/@mahijain9211/license-plate-detection-from-video-files-using-yolo-and-easyocr-6b647f0c94d5

Source for model referenced in original article (don't use that model) https://huggingface.co/Snearec/detectorMalezasYolo8/blob/2332b15b097b3f9f94fc5a260d59dae1e1b8c443/best_float32.tflite

Source for alternative model that does a better job: https://github.com/sveyek/Video-ANPR

Adapted script to receive command line arguments
https://www.tutorialspoint.com/python/python_command_line_arguments.htm

And added some extra output besides the video.


## alternative implementations

https://www.geeksforgeeks.org/python/detect-and-recognize-car-license-plate-from-a-video-in-real-time/

https://github.com/wavelolz/Video-License-Plate-Recognition

https://github.com/sveyek/Video-ANPR

