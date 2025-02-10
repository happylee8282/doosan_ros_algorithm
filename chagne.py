import os
import shutil
import time
import random

def split_files(source_directory, 
                train_image_dir="./aml_data/train/images", 
                validation_image_dir="./aml_data/valid/images", 
                test_image_dir="./aml_data/test/images",
                train_label_dir="./aml_data/train/labels",
                validation_label_dir="./aml_data/valid/labels",
                test_label_dir="./aml_data/test/labels",
                train_ratio=0.6, validation_ratio=0.2):
    """
    Splits image and label files in the source directory into train, validation, and test directories.

    Parameters:
    - source_directory: Directory containing files to split.
    - train_image_dir: Directory for training image files.
    - validation_image_dir: Directory for validation image files.
    - test_image_dir: Directory for test image files.
    - train_label_dir: Directory for training label files.
    - validation_label_dir: Directory for validation label files.
    - test_label_dir: Directory for test label files.
    - train_ratio: Proportion of files to be used for training.
    - validation_ratio: Proportion of files to be used for validation.
    """

    # Ensure ratios sum to 1
    if (train_ratio + validation_ratio) > 1:
        print("Error: Ratios sum to more than 1.")
        return

    # Create destination directories if they do not exist
    for dir_name in [train_image_dir, validation_image_dir, test_image_dir,
                     train_label_dir, validation_label_dir, test_label_dir]:
        os.makedirs(dir_name, exist_ok=True)

    # Get all files in the source directory
    all_files = [f for f in os.listdir(source_directory) if os.path.isfile(os.path.join(source_directory, f))]

    # Separate files into images and labels
    image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    label_files = [f for f in all_files if f.lower().endswith('.txt')]

    random.seed(time.time())
    random.shuffle(image_files)  # Shuffle to randomize file selection

    # Calculate split indices
    total_images = len(image_files)
    train_end = int(total_images * train_ratio)
    validation_end = train_end + int(total_images * validation_ratio)

    # Split image files
    train_image_files = image_files[:train_end]
    validation_image_files = image_files[train_end:validation_end]
    test_image_files = image_files[validation_end:]

    # Find corresponding label files
    def find_label_file(image_file):
        base_name = os.path.splitext(image_file)[0]
        label_file = f"{base_name}.txt"
        return label_file if label_file in label_files else None

    train_label_files = [find_label_file(f) for f in train_image_files if find_label_file(f)]
    validation_label_files = [find_label_file(f) for f in validation_image_files if find_label_file(f)]
    test_label_files = [find_label_file(f) for f in test_image_files if find_label_file(f)]

    # Function to copy files to destination directories
    def copy_files(files, destination):
        for file in files:
            shutil.copy(os.path.join(source_directory, file), os.path.join(destination, file))

    # Copy image and label files to their respective directories
    copy_files(train_image_files, train_image_dir)
    copy_files(validation_image_files, validation_image_dir)
    copy_files(test_image_files, test_image_dir)

    copy_files(train_label_files, train_label_dir)
    copy_files(validation_label_files, validation_label_dir)
    copy_files(test_label_files, test_label_dir)

    print(f"Files split into train, validation, and test directories.")


def main():
    source_directory = r'/home/happy/Desktop/car_by_amr'
    split_files(source_directory)

if __name__ == "__main__":
    main()
