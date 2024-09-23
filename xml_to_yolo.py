# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 12:32:36 2024

@author: efm-workstation
"""

import os
import xml.etree.ElementTree as ET

# Classes in your dataset
classes = ['Auto', 'Bus', 'Tempo', 'Tractor', 'Truck']

# Path where the XML annotations are located
annotation_base_dir = 'D:/New folder/annotation/annotation'
# Path where images are located
image_base_dir = 'D:/New folder/Vehicle_classes'
# Path where YOLO labels will be saved
label_base_dir = 'D:/New folder/labels'

# Create directory for labels if it doesn't exist
os.makedirs(label_base_dir, exist_ok=True)

# Function to convert XML annotations to YOLO format
def convert_xml_to_yolo(xml_folder, label_folder):
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(xml_folder, xml_file)
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # Image dimensions
                img_width = int(root.find('size/width').text)
                img_height = int(root.find('size/height').text)

                # Create the YOLO label file
                label_file = os.path.join(label_folder, xml_file.replace('.xml', '.txt'))
                with open(label_file, 'w') as f:
                    for obj in root.findall('object'):
                        class_name_from_xml = obj.find('name').text
                        
                        # Standardize the class names to match your defined list
                        if class_name_from_xml.lower() == 'auto':
                            class_name_from_xml = 'Auto'
                        elif class_name_from_xml.lower() == 'bus':
                            class_name_from_xml = 'Bus'
                        # Add more elif conditions for other classes if needed

                        # Check if the object class exists in the predefined classes
                        if class_name_from_xml not in classes:
                            print(f"Skipping class {class_name_from_xml} as it's not in the predefined classes.")
                            continue  # Skip objects that aren't in the classes list

                        # Get the class ID for YOLO format
                        class_id = classes.index(class_name_from_xml)

                        # Get bounding box coordinates
                        xml_box = obj.find('bndbox')
                        xmin = float(xml_box.find('xmin').text)
                        ymin = float(xml_box.find('ymin').text)
                        xmax = float(xml_box.find('xmax').text)
                        ymax = float(xml_box.find('ymax').text)

                        # Debugging print statements
                        print(f"Processing {xml_file}:")
                        print(f"Class: {class_name_from_xml}, BBox: ({xmin}, {ymin}), ({xmax}, {ymax})")

                        # Convert to YOLO format (normalized values between 0 and 1)
                        x_center = (xmin + xmax) / 2.0 / img_width
                        y_center = (ymin + ymax) / 2.0 / img_height
                        width = (xmax - xmin) / img_width
                        height = (ymax - ymin) / img_height

                        # Write to the label file in YOLO format
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                print(f"Converted: {xml_file}")
            except Exception as e:
                print(f"Error processing {xml_file}: {e}")

# Process all classes in the dataset
for class_name in classes:
 # Path to images and annotations for this class
     image_dir = os.path.join(image_base_dir, class_name)
     annotation_dir = os.path.join(annotation_base_dir, class_name)
     label_dir = os.path.join(label_base_dir, class_name)

 # Ensure label directory exists for each class
 os.makedirs(label_dir, exist_ok=True)

 # Convert XML annotations to YOLO format for this class
 convert_xml_to_yolo(annotation_dir, label_dir)

print("Conversion for all classes completed.")