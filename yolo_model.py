import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
import numpy as np
import os

class YoloHeatMap:
    def __init__(self, model, video_height, video_width, num_rows, num_cols, cell_size):
        self.yolo = YOLO(model)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.video_height = video_height
        self.video_width = video_width
        self.heat_matrix = np.zeros((self.num_rows, self.num_cols))
        self.cell_size = cell_size
        self.trajectories = []  # Store trajectories of detected objects
        
        # Assign a random color to each object class (80 classes by default)
        self.COLORS = np.random.uniform(0, 255, size=(80, 3))
        self.object_colors = {}  # Dictionary to store color per object

    def detection(self, is_show=False, image=None):
        result = self.yolo.predict(show=is_show, source=image, classes=[0])

        boxes = []
        classes = []
        names = []
        confidence = []
        for re in result:
            box = re.boxes.xyxy.tolist()
            clss = re.boxes.cls.tolist()
            name = [re.names[int(c)] if c == 0.0 else 'unknown' for c in clss]
            conf = re.boxes.conf.tolist()

            boxes.append(box)
            classes.append(clss)
            names.append(name)
            confidence.append(conf)

        return boxes, classes, names, confidence

    def draw_prediction(self, image, x_top, y_top, x_bottom, y_bottom, clss, name):
        # Assign a specific color for each class type (person, car, etc.)
        if clss not in self.object_colors:
            self.object_colors[clss] = self.COLORS[int(clss)]

        # Directly use the color without normalization
        color = self.object_colors[clss].astype(int).tolist()  # Use color in [0, 255] range
        cv2.rectangle(image, (x_top, y_top), (x_bottom, y_bottom), color, 2)
        cv2.putText(image, name, (x_top, y_top), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def draw_grid(self, image):
        for i in range(self.num_rows):
            start_point = (0, (i + 1) * self.cell_size)
            end_point = (self.video_width, (i + 1) * self.cell_size)
            color = (255, 255, 255)
            thickness = 1
            image = cv2.line(image, start_point, end_point, color, thickness)

        for i in range(self.num_cols):
            start_point = ((i + 1) * self.cell_size, 0)
            end_point = ((i + 1) * self.cell_size, self.video_height)
            color = (255, 255, 255)
            thickness = 1
            image = cv2.line(image, start_point, end_point, color, thickness)

        return image

    def heat_increase(self, x_top, y_top, x_bottom, y_bottom):
        y_center = (y_top + y_bottom) // 2
        x_center = (x_top + x_bottom) // 2
        row = y_center // self.cell_size
        col = x_center // self.cell_size
        row = min(max(row, 0), self.num_rows - 1)
        col = min(max(col, 0), self.num_cols - 1)
        self.heat_matrix[row, col] += 1
        
        # Save the center of the bounding box to trajectories
        self.trajectories.append((x_center, y_center))

    def save_trajectory(self, output_path):
        if self.trajectories:
            x, y = zip(*self.trajectories)
            plt.figure(figsize=(10, 6))
            
            # Normalize colors and plot only the last point of each trajectory
            for i, trajectory in enumerate(self.trajectories):
                x_coords, y_coords = zip(*trajectory)
                
                # Get the last point of the trajectory
                last_x, last_y = x_coords[-1], y_coords[-1]
                
                # Normalize color to the range [0, 1]
                color = self.object_colors[i % len(self.object_colors)] / 255.0
                
                # Plot only the last point with larger marker size
                plt.plot(last_x, last_y, marker='o', markersize=5, linestyle='-', color=color)

            plt.title('Final Object Positions')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.xlim(0, self.num_cols * self.cell_size)
            plt.ylim(0, self.num_rows * self.cell_size)
            plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
            plt.grid()
            plt.savefig(output_path)
            plt.close()
