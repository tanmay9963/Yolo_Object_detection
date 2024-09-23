# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 22:28:23 2024

@author: efm-workstation
"""
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


# Define directories
image_base_dir = 'D:/IEEE/image_dataset' # Base directory for class-specific image folders
label_base_dir = 'D:/IEEE/labels' # Base directory for class-specific label folders
train_image_dir = 'D:/IEEE/Vehicle_classes/images/train'
val_image_dir = 'D:/IEEE/Vehicle_classes/images/val'
train_label_dir = 'D:/IEEE/Vehicle_classes/labels/train'
val_label_dir = 'D:/IEEE/Vehicle_classes/labels/val'

# Create directories if not exist
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# List of all classes
classes = ['Auto', 'Bus', 'Tempo', 'Tractor', 'Truck']

# Function to move files (images and labels) to the respective train/val directories
def move_files(image_list, class_name, image_src, label_src, image_dst, label_dst):
 for img_name in image_list:
 # Move image
     src_img_path = os.path.join(image_src, img_name)
     dst_img_path = os.path.join(image_dst, class_name + "_" + img_name) # Rename image to avoid conflicts
     shutil.copy(src_img_path, dst_img_path)
    
     # Move corresponding label
     label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
     src_label_path = os.path.join(label_src, label_name)
     dst_label_path = os.path.join(label_dst, class_name + "_" + label_name) # Rename label to match the new image name
     shutil.copy(src_label_path, dst_label_path)

# Process each class
for class_name in classes:
 print(f"Processing class: {class_name}")
 
 # Directories for images and labels of the current class
 image_dir = os.path.join(image_base_dir, class_name)
 label_dir = os.path.join(label_base_dir, class_name)

 # List all images and corresponding labels
 images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

 # Check for images with corresponding labels
 images_with_labels = []
 for img in images:
     label_path = os.path.join(label_dir, img.replace('.jpg', '.txt').replace('.png', '.txt'))
     if os.path.exists(label_path):
         images_with_labels.append(img)
         else:
             print(f"Label not found for image: {img} in class {class_name}")
            
             # Skip if no images with labels are found
             if not images_with_labels:
             print(f"No images with corresponding labels found for class {class_name}.")
             continue

 # Split into train and validation sets
 train_images, val_images = train_test_split(images_with_labels, test_size=0.2, random_state=42)

 # Move training files
 move_files(train_images, class_name, image_dir, label_dir, train_image_dir, train_label_dir)

 # Move validation files
 move_files(val_images, class_name, image_dir, label_dir, val_image_dir, val_label_dir)

print("Train and validation split completed for all classes!")




model = YOLO("yolov8n.pt") # Replace 'yolov8n.pt' with any other model if needed

# Path to the YAML file that defines the dataset
dataset_yaml = "D:/IEEE/Vehicle_classes/vehicle_dataset.yaml"

# Train the YOLO model on your custom dataset
model.train(data='D:/IEEE/Vehicle_classes/vehicle_dataset.yaml', epochs=100, imgsz=640, batch=16, freeze=[0])


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