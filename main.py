import cv2
from ultralytics import YOLO
import easyocr
from PIL import Image
import numpy as np

# added for arguments parsing 
import argparse
parser=argparse.ArgumentParser(description="License Plate Scanner")
parser.add_argument("sourcePath", nargs='?', default="input.mp4")
parser.add_argument("outputPath", nargs='?', default="output.mp4")
parser.add_argument("skipFrames", nargs='?', type=int, default="1")
parser.add_argument("resHorizontal", nargs='?', type=int, default="640")
parser.add_argument("resVertical", nargs='?', type=int, default="480")
parser.add_argument("outputCsvPath", nargs='?', default="output.csv")
parser.add_argument("confidenceLimit", nargs='?', type=float, default="0.1")

args=parser.parse_args()

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Load your YOLO model (replace with your model's path)
#model = YOLO('./models/yolov8n.pt', task='detect')
model = YOLO('./models/license_plate_detector.pt', task='detect')

# Open the video file (replace with your video file path)
video_path = args.sourcePath #'input.mp4'
cap = cv2.VideoCapture(video_path)

# Create a VideoWriter object (optional, if you want to save the output)
output_path = args.outputPath #'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
resHorizontal = args.resHorizontal #640
resVertical = args.resVertical #480
out = cv2.VideoWriter(output_path, fourcc, 30.0, (resHorizontal, resVertical))  # Adjust frame size if necessary

# Frame skipping factor (adjust as needed for performance)
frame_skip = args.skipFrames #3 # Skip every 3rd frame
frame_count = 0

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

    # Resize the frame (optional, adjust size as needed)
    frame = cv2.resize(frame, (resHorizontal, resVertical))  # Resize to 640x480

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

                # Use EasyOCR to read text from plate
                plate_number = reader.readtext(plate_array)
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

                # collect results
                if( not np.isnan(number_conf) ):
                    if ( number_conf > args.confidenceLimit ):
                        plates.append([frame_count, concat_number, f"{number_conf:.2f}"])

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

# for text (csv) based output
import csv
with open(args.outputCsvPath, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')#, quotechar='', quoting=csv.QUOTE_MINIMAL)
    # create some heading
    csvwriter.writerow(["Video Frame","License Plate","Confidence"])
    csvwriter.writerows(plates)

# Release resources
cap.release()
out.release()  # Release the VideoWriter object if used
cv2.destroyAllWindows()
