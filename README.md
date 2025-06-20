# Video License Plate Recognition

A python script that takes a video mp4 file as input and checks for license plates frame by frame
resulting in video output with text based info overlay and a CSV text file. 

<img src="https://github.com/hanscappelle/video-license-plate-recognition/blob/main/frame-309.JPG"/>

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

Optional step for mac install metal for gpu support
```
pip3.11 install tensorflow-metal
```

## Execute this code

Find out what arguments are supported:

```
hcpl@Hanss-MacBook-Air video-license-plate-recognition % Python3.11 main.py -h
usage: main.py [-h] [sourceFile] [outputPath] [rotate180] [skipFrames] [confidenceLimit] [exportFrames] [resHorizontal] [resVertical] [outVideoFile] [outCsvFile]

License Plate Scanner

positional arguments:
  sourceFile        # source for original input video
  outputPath        # path for output (csv, frame and overlay video output)
  rotate180         # 1 to rotate video 180d
  skipFrames        # the number of frames to skip 
  confidenceLimit   # confidence limit (0.0 - 1.0)
  exportFrames      # 1 to export detected frames 
  resHorizontal     # optional horizontal resolution for resize
  resVertical       # optional vertical resolution for resize
  outVideoFile      # output video name, defaults to output.mp4
  outCsvFile        # output csv file name, defaults to output.csv

options:
  -h, --help       show this help message and exit
```

Example for skipping every 3 frames working with a downsampled video in 1280x960 resolution 
and output text only if confidence is above 0.3.

```
python3.11 main.py input.mp4 output 0 3 0.3 1 1280 960
```

Example output running above command

```
(venv) hcpl@Mac Video-LPR % python3 main.py         
Using CPU. Note: This module is much faster with a GPU.
Downloading detection model, please wait. This may take several minutes depending upon your network connection.
Progress: |██████████████████████████████████████████████████| 100.0% CompleteDownloading recognition model, please wait. This may take several minutes depending upon your network connection.
Progress: |██████████████████████████████████████████████████| 100.0% Complete% (venv) hcpl@Mac Video-LPR % 
```

## Troubleshooting

Most common issues:
- using a non compatible Python version, stick to a version within the `3.9-11` range 
- swapped arguments, double check the order of arguments using the help option `-h`
- using cpu instead of gpu will slow down performance

## Performance

### Speed

I'm working on a Mac M1 myself so for GPU support to be enabled I had to install tensorflow-metal. 
You can force gpu or cpu in code if needed.  

Frame skipping can be used to reduce the total number of frames of the video that should be processed. 
For example setting this to 3 will jump over and process only every 3 frames of the video.

Limited processing to recognized rectangles is a must for speed and performance, otherwise opencv
will detect all objects it finds on the road and even categorise like cars, trucks, ...

For that the included model works well:
```
./models/license_plate_detector.pt
```

Video frame size can be reduced to speed up OCR per frame. For example a 4K video with original
resolution of 3840 * 2160 can be processed as a (3840/3=) 1280 * (2160/3=) 720.

### Precision

However reducing frame size I did see a big decrease in correctly recognized license plates. The 
rectangles are still found but the OCR part suffers. For example from that same 4K image source
of original 3840x2160 size the full size image results in these (not complete):

```
Video Frame,License Plate,Confidence
165,  2-EXH-885,  0.41
177,  2EX885,     0.54
234,  24FK-92,    0.40
237,  2-FV-92,    0.47
243,  24,         0.78
294,  LD-401,     0.49
300,  2-AUD-41,   0.84
306,  2-AUD-40,   0.63
309,  2-AUD-401,  0.96
```

While the same first hits for the video when size was reduced by 1/3 (see prev) looks like this:

```
Video Frame,License Plate,Confidence
21,   1 RDB 459,    0.88
24,   1 RDB L59,    0.53
27,   1 RDB 459,    0.87
30,   1 RDB 159,    0.71
33,   RDB 653,      0.56
66,   2 AFL 743,    0.96
69,   4FL 743,      0.63
135,  2PEGK 73,     0.98
138,  2EEGK 731,    0.63
```

## Improvements

* don't wait till the end to export csv text based results
* improve models? For license plate and for OCR (seem to perform OK)
* better way of confiugration needed, order of parameters is confusing

### Done

* add parameters for more options
* also output text based in CSV format
* limiting to allowed characters with
  allowlist (string) - Force EasyOCR to recognize only subset of characters. Useful for specific problem (E.g. license plate, etc.)
  from https://www.jaided.ai/easyocr/documentation/
  0123456789-ABCDEFGHIJKLMNOPQRSTUVWXYZ
* also write detected frames https://roboflow.com/use-opencv/save-an-image-with-imwrite
* use original frame size by default
* optional 180 degrees rotation option
* it probably doesn't help that my source video was anamorphic and desqueezed, tested with normal lens & 5K, 
  5K resolution is overkill, you'll achieve better results getting closer to the license plates instead

## References

Source for original version of this code found in this article (see _main.py in this project for reference): 
https://medium.com/@mahijain9211/license-plate-detection-from-video-files-using-yolo-and-easyocr-6b647f0c94d5

Source for model referenced in original article (don't use that model): 
https://huggingface.co/Snearec/detectorMalezasYolo8/blob/2332b15b097b3f9f94fc5a260d59dae1e1b8c443/best_float32.tflite

Source for alternative model that does a way better job: 
https://github.com/sveyek/Video-ANPR

Adapted script to receive command line arguments using this info:
https://www.tutorialspoint.com/python/python_command_line_arguments.htm

https://medium.com/@adityamahajan.work/easyocr-a-comprehensive-guide-5ff1cb850168

https://medium.com/analytics-vidhya/ocr-corrector-in-regex-extraction-6d2af0d92dc

## Alternative implementations

Just some similar projects I found. 

https://www.geeksforgeeks.org/python/detect-and-recognize-car-license-plate-from-a-video-in-real-time/

https://github.com/wavelolz/Video-License-Plate-Recognition

https://github.com/sveyek/Video-ANPR

https://github.com/mendez-luisjose/License-Plate-Detection-with-YoloV8-and-EasyOCR

