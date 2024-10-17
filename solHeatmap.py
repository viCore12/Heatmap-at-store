import cv2
import time
from ultralytics import solutions

# Open video file
cap = cv2.VideoCapture("../Tracking/test_videos/cctv.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("heatmap_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize heatmap model with the person class filter
heatmap = solutions.Heatmap(
    show=True,
    model="yolo11n.pt",  # Replace with the correct path/model if needed
    colormap=cv2.COLORMAP_PARULA,
    classes=[0],  # Only detect the 'person' class (class_id = 0)
    line_width=5,
)

# Variables for FPS calculation
prev_time = time.time()
fps_display = 0

# Process the video frames
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    
    # Calculate FPS
    curr_time = time.time()
    fps_display = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    # Generate heatmap for person class only
    im0 = heatmap.generate_heatmap(im0)
    
    # Display FPS on the frame
    cv2.putText(im0, f"FPS: {fps_display:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Write the processed frame to output video
    video_writer.write(im0)

# Release resources
cap.release()
video_writer.release()
