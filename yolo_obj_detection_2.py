# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 12:48:40 2024

@author: efm-workstation
"""

import os
import shutil
import random
import os
import xml.etree.ElementTree as ET
import imgaug.augmenters as iaa
import os
import cv2
import shutil
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from ultralytics import YOLO



# Paths for images and labels base directories
image_base_dir = 'D:/IEEE/image_label_class/images'
label_base_dir = 'D:/IEEE/image_label_class/labels'

# Create directories for train and test splits
train_image_dir = 'D:/IEEE/image_label_class/train/images'
train_label_dir = 'D:/IEEE/image_label_class/train/labels'
test_image_dir = 'D:/IEEE/image_label_class/test/images'
test_label_dir = 'D:/IEEE/image_label_class/test/labels'

# Create directories if they don't exist
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)

# Define the train-test split ratio
split_ratio = 0.8  # 80% train, 20% test

# Function to split a class directory into train and test
def split_class(class_name):
    # Paths for images and labels in the class directory
    image_class_dir = os.path.join(image_base_dir, class_name)
    label_class_dir = os.path.join(label_base_dir, class_name)
    
    # Get list of all files in the class directory
    image_files = os.listdir(image_class_dir)
    
    # Shuffle the files
    random.shuffle(image_files)
    
    # Split files into train and test sets
    split_index = int(len(image_files) * split_ratio)
    train_files = image_files[:split_index]
    test_files = image_files[split_index:]
    
    # Create train and test directories for this class
    train_image_class_dir = os.path.join(train_image_dir, class_name)
    train_label_class_dir = os.path.join(train_label_dir, class_name)
    test_image_class_dir = os.path.join(test_image_dir, class_name)
    test_label_class_dir = os.path.join(test_label_dir, class_name)
    
    os.makedirs(train_image_class_dir, exist_ok=True)
    os.makedirs(train_label_class_dir, exist_ok=True)
    os.makedirs(test_image_class_dir, exist_ok=True)
    os.makedirs(test_label_class_dir, exist_ok=True)
    
    # Move train files
    for file_name in train_files:
        # Move image
        src_image_path = os.path.join(image_class_dir, file_name)
        dst_image_path = os.path.join(train_image_class_dir, file_name)
        shutil.copy(src_image_path, dst_image_path)
        
        # Move corresponding label
        label_file_name = file_name.replace('.jpg', '.txt')  # Assuming .jpg for images and .txt for labels
        src_label_path = os.path.join(label_class_dir, label_file_name)
        dst_label_path = os.path.join(train_label_class_dir, label_file_name)
        shutil.copy(src_label_path, dst_label_path)
    
    # Move test files
    for file_name in test_files:
        # Move image
        src_image_path = os.path.join(image_class_dir, file_name)
        dst_image_path = os.path.join(test_image_class_dir, file_name)
        shutil.copy(src_image_path, dst_image_path)
        
        # Move corresponding label
        label_file_name = file_name.replace('.jpg', '.txt')
        src_label_path = os.path.join(label_class_dir, label_file_name)
        dst_label_path = os.path.join(test_label_class_dir, label_file_name)
        shutil.copy(src_label_path, dst_label_path)
    
    print(f"Class '{class_name}' has been split into train and test sets.")

# Iterate through all classes and split
classes = os.listdir(image_base_dir)
for class_name in classes:
    split_class(class_name)

print("Train-test split completed for all classes.")



model = YOLO("yolov8n.pt")

# Path to the YAML file that defines the dataset
dataset_yaml = "D:/IEEE/image_label_class/yolo_dataset.yaml"

# Train the YOLO model on your custom dataset
model.train(data='D:/IEEE/image_label_class/yolo_dataset.yaml', epochs=100, imgsz=640, batch=16, freeze=[0])


# Save the trained model
model.save("D:/IEEE/trained_model.pt")

# Validate the model (optional step after training)
model.val(data=dataset_yaml)



# Load the trained YOLO model
model = YOLO("D:/IEEE/trained_model.pt")  # Path to your trained model

# Video file paths
video_path = "D:/IEEE/18th_Crs_BsStp_JN_FIX_1_time_2024-05-12T12_30_02_000.mp4"  # Input video path
output_video_path = "D:/IEEE/output_video.mp4"  # Path to save the output video
csv_output_path = "D:/IEEE/predictions_video.csv"  # Path to save the predictions as a CSV file

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video details (width, height, frames per second)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Initialize a list to store predictions
predictions = []

# Process each frame of the video
while True:
    ret, frame = cap.read()  # Read each frame
    if not ret:
        break

    # Get timestamp in seconds
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    # Run YOLO model on the frame
    results = model(frame)

    # Draw bounding boxes on the frame
    annotated_frame = results[0].plot()

    # Write the frame to the output video
    out.write(annotated_frame)

    # Extract prediction data (class ID, bounding boxes, confidence)
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        x_center = box.xywh[0][0].item()
        y_center = box.xywh[0][1].item()
        box_width = box.xywh[0][2].item()
        box_height = box.xywh[0][3].item()
        confidence = box.conf[0].item()
        
        # Append the prediction data for this frame
        predictions.append([video_path, timestamp, class_id, x_center, y_center, box_width, box_height, confidence])

# Release resources
cap.release()
out.release()

# Save predictions to a CSV file
df = pd.DataFrame(predictions, columns=['video_path', 'timestamp', 'class_id', 'x_center', 'y_center', 'width', 'height', 'confidence'])
df.to_csv(csv_output_path, index=False)

print(f"Video processing complete. Saved output video to: {output_video_path} and CSV to: {csv_output_path}")


df=pd.read_csv("predictions_video.csv")
df.head(5)

df.shape
df[['timestamp','class_id']]
df['class_id'].value_counts()

classes = ['Auto', 'Bus', 'Tempo', 'Tractor', 'Truck']
class_id_name = {i:name for i, name in enumerate(classes) }
df['classes']=df['class_id'].map(class_id_name)
df.columns
df[['classes','class_id']]

df['class_id']


metrics = model.val(data=dataset_yaml)