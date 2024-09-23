import os
import requests
from tqdm import tqdm
from pycocotools.coco import COCO
import zipfile
import json

# Directories for saving images and annotations
output_dir = "filtered_coco_dataset"
annotations_dir = os.path.join(output_dir, "annotations")
class_dirs = {
    'person': os.path.join(output_dir, 'images/person'),
    'car': os.path.join(output_dir, 'images/car'),
    'motorcycle': os.path.join(output_dir, 'images/motorcycle'),
    'bus': os.path.join(output_dir, 'images/bus')
}

# Create directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)
for class_dir in class_dirs.values():
    os.makedirs(class_dir, exist_ok=True)

# COCO dataset paths
annotation_zip_file = "annotations_trainval2017.zip"
coco_annotations_file = os.path.join(output_dir, "annotations", "instances_train2017.json")

# Download COCO annotations if not already present
if not os.path.exists(coco_annotations_file):
    print("Downloading COCO annotations JSON file...")
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    response = requests.get(url, stream=True)
    with open(annotation_zip_file, "wb") as f:
        f.write(response.content)
    print("Downloaded COCO annotations ZIP.")
    
    with zipfile.ZipFile(annotation_zip_file, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    print("Extracted COCO annotations.")

# Load COCO annotations
coco = COCO(coco_annotations_file)

# Classes to download (including Bus now)
target_classes = ['person', 'car', 'motorcycle', 'bus']

# Get category IDs for the target classes
category_ids = coco.getCatIds(catNms=target_classes)
print(f"Category IDs for {target_classes}: {category_ids}")

# Function to download an image and save it to its corresponding class directory
def download_image(img_info, class_dir):
    img_url = img_info['coco_url']
    img_path = os.path.join(class_dir, img_info['file_name'])
    
    if not os.path.exists(img_path):
        response = requests.get(img_url, stream=True)
        with open(img_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Downloaded {img_info['file_name']} to {class_dir}")
    return img_path

# Download images and save them in class-specific folders, also save separate annotations
print("Downloading images for target classes...")

class_annotations = {class_name: [] for class_name in target_classes}
class_images = {class_name: [] for class_name in target_classes}

# Set the limit of 200 images per class
image_limit = 200

for class_name, class_id in zip(target_classes, category_ids):
    class_image_ids = coco.getImgIds(catIds=[class_id])[:image_limit]  # Limit to 200 images
    print(f"Downloading {len(class_image_ids)} images for class '{class_name}'")
    
    for img_id in tqdm(class_image_ids):
        img_info = coco.loadImgs(img_id)[0]
        # Download the image
        download_image(img_info, class_dirs[class_name])
        # Store image metadata for this class
        class_images[class_name].append(img_info)
        # Get and store the corresponding annotations for this class
        ann_ids = coco.getAnnIds(imgIds=[img_info['id']], catIds=[class_id])
        anns = coco.loadAnns(ann_ids)
        class_annotations[class_name].extend(anns)

# Save annotations separately for each class
print("Saving annotations for each class...")

for class_name, anns in class_annotations.items():
    # Create the full structure for the annotation file
    coco_structure = {
        "info": coco.dataset['info'],  # General dataset info
        "licenses": coco.dataset['licenses'],  # License info
        "images": class_images[class_name],  # Images metadata
        "annotations": anns,  # Annotations metadata
        "categories": [cat for cat in coco.loadCats(category_ids) if cat['name'] == class_name]  # Category info
    }
    
    # Save the annotation file in correct format
    class_annotation_file = os.path.join(annotations_dir, f'instances_train2017_{class_name}.json')
    with open(class_annotation_file, 'w') as f:
        json.dump(coco_structure, f)
    print(f"Annotations for class '{class_name}' saved to {class_annotation_file}")

print(f"Process complete. Images and annotations are separated by class in {output_dir}")





import os
import json
from pycocotools.coco import COCO

# Define directories
output_dir = "filtered_coco_dataset"
annotation_files = {
    'person': "filtered_coco_dataset/annotations/instances_train2017_person.json",
    'car': "filtered_coco_dataset/annotations/instances_train2017_car.json",
    'motorcycle': "filtered_coco_dataset/annotations/instances_train2017_motorcycle.json",
    'bus': "filtered_coco_dataset/annotations/instances_train2017_bus.json"
}
image_base_dir = os.path.join(output_dir, "images")
label_base_dir = os.path.join(output_dir, "labels")

# Create the label directory if it doesn't exist
os.makedirs(label_base_dir, exist_ok=True)

# Function to convert COCO bbox to YOLO format
def coco_to_yolo(bbox, img_width, img_height):
    x_min, y_min, box_width, box_height = bbox
    x_center = (x_min + box_width / 2) / img_width
    y_center = (y_min + box_height / 2) / img_height
    width = box_width / img_width
    height = box_height / img_height
    return x_center, y_center, width, height

# Process each annotation file
for class_name, annotation_file in annotation_files.items():
    # Load COCO annotation file
    coco = COCO(annotation_file)
    
    # Create directory for YOLO labels
    class_label_dir = os.path.join(label_base_dir, class_name)
    os.makedirs(class_label_dir, exist_ok=True)

    # Get all image IDs for the class
    image_ids = coco.getImgIds()

    for image_id in image_ids:
        # Load image info
        img_info = coco.loadImgs(image_id)[0]
        img_width = img_info['width']
        img_height = img_info['height']
        img_filename = img_info['file_name']

        # Get all annotations for this image
        ann_ids = coco.getAnnIds(imgIds=[image_id])
        anns = coco.loadAnns(ann_ids)

        # YOLO label file for the image
        label_file_path = os.path.join(class_label_dir, f"{os.path.splitext(img_filename)[0]}.txt")
        
        # Write YOLO labels
        with open(label_file_path, 'w') as label_file:
            for ann in anns:
                bbox = ann['bbox']  # COCO bbox: [x_min, y_min, width, height]
                category_id = ann['category_id']  # Class ID

                # Convert bbox to YOLO format
                x_center, y_center, width, height = coco_to_yolo(bbox, img_width, img_height)

                # YOLO format: <class_id> <x_center> <y_center> <width> <height>
                label_file.write(f"{category_id} {x_center} {y_center} {width} {height}\n")

    print(f"Converted annotations for {class_name} to YOLO format.")

print("All COCO annotations successfully converted to YOLO format.")