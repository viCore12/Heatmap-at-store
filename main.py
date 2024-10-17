import cv2
from yolo_model import YoloHeatMap
from imutils.video import FPS
from skimage.transform import resize
import numpy as np
import argparse
import time

# Argument parsing
parser = argparse.ArgumentParser(description="Process video with YOLO and heatmap")
parser.add_argument('--video', type=str, default="video.retail.mp4", help='Path to the input video file')
parser.add_argument('--model', type=str, default='yolov8m.pt', help='Path to YOLO model file')
parser.add_argument('--output', type=str, default='vid_output.output_video.mp4', help='Path to save the output video')
parser.add_argument('--heatmap_output', type=str, default='vid_output.heatmap_video.mp4', help='Path to save the heatmap video')
args = parser.parse_args()

video_file = args.video
model_file = args.model
output_file = args.output
heatmap_output_file = args.heatmap_output

alpha = 0.4
video_height = 384
video_width = 640
cell_size = 40
num_cols = video_width // cell_size
num_rows = video_height // cell_size

# Initialize YOLO model
custom_model = YoloHeatMap(model_file, video_height, video_width, num_rows, num_cols, cell_size)

# Initialize video capture and output writers
video = cv2.VideoCapture(video_file)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
fps = int(video.get(cv2.CAP_PROP_FPS))   
out = cv2.VideoWriter(output_file, fourcc, fps, (video_width, video_height))

heatmap_out = cv2.VideoWriter(heatmap_output_file, fourcc, fps, (video_width, video_height))

# Start FPS counter
fps_counter = FPS().start()

while True:
    ret, frame = video.read()
    if not ret:
        break

    start_time = time.time()  # Start time for FPS calculation

    frame = cv2.resize(frame, (video_width, video_height))

    # Perform detection with YOLO
    boxes, classes, names, _ = custom_model.detection(False, frame)
    # Draw predictions and update heatmap
    for i in range(len(boxes[0])):
        x1, y1, x2, y2 = boxes[0][i]
        clss = classes[0][i]
        name = names[0][i]
        custom_model.draw_prediction(frame, round(x1), round(y1), round(x2), round(y2), clss, name)
        custom_model.heat_increase(round(x1), round(y1), round(x2), round(y2))

    #custom_model.save_trajectory('trajectory_video.png')
    # Generate the heatmap
    temp_heat_matrix = custom_model.heat_matrix.copy()
    temp_heat_matrix = resize(temp_heat_matrix, (video_height, video_width))
    temp_heat_matrix = temp_heat_matrix / np.max(temp_heat_matrix)
    temp_heat_matrix = np.uint8(temp_heat_matrix * 255)

    # Save heatmap
    image_heat = cv2.applyColorMap(temp_heat_matrix, cv2.COLORMAP_JET)
    frame_heat = custom_model.draw_grid(image_heat) #Draw grid
    heatmap_out.write(frame_heat)
    
    # Calculate FPS and display it on the frame
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Blend the heatmap with the original frame
    #cv2.addWeighted(image_heat, alpha, frame, 1 - alpha, 0, frame)
    out.write(frame)

    # Update FPS counter
    fps_counter.update()

# Stop the FPS counter and print final stats
fps_counter.stop()

# Release resources
video.release()
out.release()
heatmap_out.release()
