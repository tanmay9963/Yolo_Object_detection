# Yolo_Object_detection
Key Details:
1. Dataset Download:

        -Indian Vehicle Dataset (Auto, Bus, Truck) from Kaggle.
        -COCO Dataset (person, car, motorcycle, bus) via COCO API.
2. Annotation Conversion:

        -Indian vehicles dataset annotations (XML) converted to YOLO format using xml_to_yolo.py.
        -COCO dataset annotations (JSON) converted to YOLO format using json_to_yolo.py.
3. Train-Test Split:

        -Splitting the data into 80% training and 20% testing using a split script.
        -Separate directories for training and testing images/labels.
4. YOLOv8 Training:

        -Train the YOLOv8 model on the custom dataset using the provided script.
        -Save the trained model for future inference.
5. Object Detection on Video:

        -Use the trained YOLO model to perform object detection on video frames.
        -Output the annotated video and save the detection results in a CSV file.
