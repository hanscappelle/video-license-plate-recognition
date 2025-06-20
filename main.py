import cv2
from ultralytics import YOLO
import easyocr
from PIL import Image
import numpy as np
import os

# added for arguments parsing 
import argparse
parser=argparse.ArgumentParser(description="License Plate Scanner")
parser.add_argument("sourcePath", nargs='?', default="input.mp4")
parser.add_argument("outputPath", nargs='?', default="output")
parser.add_argument("rotate180", nargs='?', type=int, default="0")
parser.add_argument("skipFrames", nargs='?', type=int, default="1")
parser.add_argument("confidenceLimit", nargs='?', type=float, default="0.1")
parser.add_argument("exportFrames", nargs='?', type=int, default="0")
parser.add_argument("resHorizontal", nargs='?', type=int, default="0")
parser.add_argument("resVertical", nargs='?', type=int, default="0")
parser.add_argument("outVideoFile", nargs='?', default="output.mp4")
parser.add_argument("outCsvFile", nargs='?', default="output.csv")
args=parser.parse_args()

# create output folder when needed
if not os.path.exists(args.outputPath):
    print(f'output folder created {args.outputPath}')
    os.makedirs(args.outputPath)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Load your YOLO model (replace with your model's path)
# very common model will slow down processing and recognize everything, not just plates
#model = YOLO('./models/yolov8n.pt', task='detect')
# license plate specific model
modelPath = './models/license_plate_detector.pt'
model = YOLO(modelPath, task='detect')
print(f'using modelPath {modelPath}')

# Open the video file (replace with your video file path)
video_path = args.sourcePath #'input.mp4'
cap = cv2.VideoCapture(video_path)

# OPTIONAL: check if user has preferred dimensions set
resHorizontal = args.resHorizontal #640
resVertical = args.resVertical #480
shouldResize = 1
if resHorizontal == 0 or resVertical == 0:
    print(f'frame size was not set, using frame size of source video')
    resHorizontal = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #cv2.cv.CV_CAP_PROP_FRAME_WIDTH)   # float `width`
    resVertical = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)  # float `height`
    shouldResize = 0
    print(f'found frame size {resHorizontal} x {resVertical}')

# Create a VideoWriter object (optional, if you want to save the output)
output_path = f"{args.outputPath}/{args.outVideoFile}" #'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (resHorizontal, resVertical))  # Adjust frame size if necessary
print(f'writing output video to {output_path}')

# Frame skipping factor (adjust as needed for performance)
frame_skip = args.skipFrames #3 # Skip every 3rd frame
frame_count = 0
print(f'skipping every {args.skipFrames} frame')

# collect results
plates = []

while cap.isOpened():

    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        break  # Exit loop if there are no frames left

    # Skip frames
    if frame_count % frame_skip != 0:
        frame_count += 1
        continue  # Skip processing this frame

    if shouldResize == 1:
        # Resize the frame (optional, adjust size as needed)
        frame = cv2.resize(frame, (resHorizontal, resVertical))  # Resize to 640x480

    if args.rotate180 == 1:
        # Optional: rotate 180
        frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Make predictions on the current frame
    results = model.predict(source=frame)

    # Iterate over results and draw predictions
    for result in results:
        boxes = result.boxes  # Get the boxes predicted by the model
        for box in boxes:
            class_id = int(box.cls)  # Get the class ID
            confidence = box.conf.item()  # Get confidence score
            coordinates = box.xyxy[0]  # Get box coordinates as a tensor

            # Extract and convert box coordinates to integers
            x1, y1, x2, y2 = map(int, coordinates.tolist())  # Convert tensor to list and then to int

            # Draw the box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle
            
            # Try to apply OCR on detected region
            try:
                # Ensure coordinates are within frame bounds
                r0 = max(0, x1)
                r1 = max(0, y1)
                r2 = min(frame.shape[1], x2)
                r3 = min(frame.shape[0], y2)

                # Crop license plate region
                plate_region = frame[r1:r3, r0:r2]

                # Convert to format compatible with EasyOCR
                plate_image = Image.fromarray(cv2.cvtColor(plate_region, cv2.COLOR_BGR2RGB))
                plate_array = np.array(plate_image)

                # Use EasyOCR to read text from plate, no limit on characters
                #plate_number = reader.readtext(plate_array)
                # using allowlist to limit characters in output
                plate_number = reader.readtext(plate_array, allowlist="0123456789-ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                concat_number = ' '.join([number[1] for number in plate_number])
                number_conf = np.mean([number[2] for number in plate_number])

                # Draw the detected text on the frame
                cv2.putText(
                    img=frame,
                    text=f"[{concat_number}]({number_conf:.2f})",
                    org=(r0, r1 - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    color=(0, 0, 255),
                    thickness=2
                )

                # OPTION: collect results
                if( not np.isnan(number_conf) and number_conf > args.confidenceLimit ):
                    plates.append([frame_count, concat_number, f"{number_conf:.2f}"])

                # OPTION: also store frames with detection as image
                if( args.exportFrames and number_conf > args.confidenceLimit ):
                    cv2.imwrite(f"{args.outputPath}/frame-{frame_count}.JPG", frame)

            except Exception as e:
                print(f"OCR Error: {e}")
                pass

    # Show the frame with detections
    cv2.imshow('Detections', frame)

    # Write the frame to the output video (optional)
    out.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit loop if 'q' is pressed

    frame_count += 1  # Increment frame count

# csv output

# OPTION: for text (csv) based output
import csv
with open(f"{args.outputPath}/{args.outCsvFile}", 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')#, quotechar='', quoting=csv.QUOTE_MINIMAL)
    # create some heading
    csvwriter.writerow(["Video Frame","License Plate","Confidence"])
    csvwriter.writerows(plates)

# Release resources
cap.release()
out.release()  # Release the VideoWriter object if used
cv2.destroyAllWindows()
