import os

# Define the correct classes and the correct order
new_classes = ['person', 'car', 'motorcycle', 'Auto', 'Bus', 'Tempo', 'Tractor', 'Truck']

# Directories for labels
base_label_dir = 'D:/IEEE/image_label_class/labels'

# Function to re-map class IDs explicitly
def remap_class_ids_explicit(base_label_dir, new_class_mapping):
    # Loop through each class folder inside the base label directory
    for class_folder in os.listdir(base_label_dir):
        class_folder_path = os.path.join(base_label_dir, class_folder)

        # Check if this is a directory
        if os.path.isdir(class_folder_path):
            # Go through all label (.txt) files in this folder
            for label_file in os.listdir(class_folder_path):
                if label_file.endswith('.txt'):
                    label_file_path = os.path.join(class_folder_path, label_file)
                    new_content = []

                    # Open the label file and read the contents
                    with open(label_file_path, 'r') as f:
                        lines = f.readlines()

                        # Process each line in the label file
                        for line in lines:
                            elements = line.strip().split()
                            old_class_id = int(elements[0])

                            # The folder name corresponds to the class name
                            current_class_name = class_folder

                            # Find the new class ID based on the new_classes list
                            if current_class_name in new_class_mapping:
                                new_class_id = new_class_mapping.index(current_class_name)
                                
                                # Replace the old class ID with the new class ID
                                elements[0] = str(new_class_id)
                                
                                # Prepare the updated content
                                new_content.append(" ".join(elements))

                    # Write the updated content back into the label file
                    with open(label_file_path, 'w') as f:
                        f.write("\n".join(new_content))

                    print(f"Updated class IDs in {label_file}")

# Call the function to remap the class IDs
remap_class_ids_explicit(base_label_dir, new_classes)

print("Re-mapping of class IDs completed successfully.")