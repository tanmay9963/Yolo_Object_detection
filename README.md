# Yolo_Object_detection
------------------------------------------------------+
                                    |          Download and Prepare Datasets                |
                                    +-------------------------------------------------------+
                                    |  - Download Indian Vehicle Dataset (from Kaggle)       |
                                    |  - Download COCO Dataset for Classes                  |
                                    |    ('person', 'car', 'motorcycle', 'bus')             |
                                    +-------------------------------------------------------+
                                                        |
                                                        v
+---------------------------------------------------+         +---------------------------------------------------+
|   Convert XML to YOLO Format for Indian Vehicles  |         |  Convert JSON to YOLO Format for COCO Dataset      |
|   (Auto, Bus, Truck) using 'xml_to_yolo.py'       |         |  (person, car, motorcycle, bus) using 'json_to_yolo.py'|
+---------------------------------------------------+         +---------------------------------------------------+
                                                        | 
                                                        v 
                                    +-------------------------------------------------------+
                                    |          Train-Test Split                             |
                                    +-------------------------------------------------------+
                                    |  - Split Images and Labels into Train and Test        |
                                    |  - Create Directories for Each Class                  |
                                    |  - Save Train and Test Data into Respective Folders   |
                                    +-------------------------------------------------------+
                                                        |
                                                        v
                                    +-------------------------------------------------------+
                                    |      Load YOLOv8 Pre-trained Model (yolov8n.pt)       |
                                    +-------------------------------------------------------+
                                    |  - Train YOLOv8 on Custom Dataset                    |
                                    |  - Save Trained Model                                 |
                                    +-------------------------------------------------------+
                                                        |
                                                        v
                                    +-------------------------------------------------------+
                                    |              Object Detection                        |
                                    +-------------------------------------------------------+
                                    |  - Run Object Detection on Video                     |
                                    |  - Save Output Video with Annotations                |
                                    |  - Save Predictions in CSV File
