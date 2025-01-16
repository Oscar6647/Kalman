import cv2
from ultralytics import YOLO


model = YOLO("runs/detect/train2/weights/best.pt")


cap = cv2.VideoCapture('Screw22.mp4')
output_video_path = "output_video.mp4"
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the video codec and create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model.predict(frame, imgsz=640)  # Adjust imgsz as needed

    # Draw bounding boxes and labels on the frame
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video
    out.write(annotated_frame)

# Release the video objects
cap.release()
out.release()

print("Video processing complete. Saved as", output_video_path)