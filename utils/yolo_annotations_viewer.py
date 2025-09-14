# Viewer written by omrode1
# https://github.com/omrode1/YOLO-Annotation-Viewer

import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import Counter

def load_classes(folder_path):
    """Load the class names from the classes.txt file."""
    classes_file = os.path.join(folder_path, 'classes.txt')
    with open(classes_file, 'r') as f:
        classes = f.read().splitlines()
    return classes

def load_annotations(folder_path):
    """Load YOLO annotations from the folder."""
    annotations = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt') and filename != 'classes.txt':  # Ignore classes.txt
            img_id = os.path.splitext(filename)[0]
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as f:
                annotations[img_id] = [line.strip().split() for line in f.readlines()]
    return annotations

def load_image(folder_path, image_id):
    """Load an image from the image folder."""
    for ext in ['.jpg', '.png', '.jpeg']:
        img_path = os.path.join(folder_path, image_id + ext)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            return img
    return None

def plot_annotations(ax, image, annotation_data, classes, colors):
    """Plot image with YOLO bounding boxes using Matplotlib."""
    ax.clear()  # Clear previous image and annotations
    img_height, img_width = image.shape[:2]
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for ann in annotation_data:
        class_id, x_center, y_center, width, height = map(float, ann)
        class_id = int(class_id)

        # Convert YOLO format to bounding box
        x1 = int((x_center - width / 2) * img_width)
        y1 = int((y_center - height / 2) * img_height)
        bbox_width = int(width * img_width)
        bbox_height = int(height * img_height)

        # Draw the rectangle
        rect = Rectangle((x1, y1), bbox_width, bbox_height, linewidth=2,
                         edgecolor=colors[class_id], facecolor='none')
        ax.add_patch(rect)

        # Add class label
        ax.text(x1, y1 - 5, classes[class_id], color=colors[class_id], fontsize=12, weight='bold')

    plt.draw()  # Update the plot without blocking

def count_class_instances(annotations, classes):
    """Count the instances of each class."""
    class_counts = Counter()
    for img_id, anns in annotations.items():
        for ann in anns:
            class_id = int(ann[0])
            class_counts[classes[class_id]] += 1
    return class_counts

def plot_class_distribution(class_counts):
    """Plot a bar graph for class distribution."""
    labels = list(class_counts.keys())
    counts = list(class_counts.values())
    plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Number of Instances')
    plt.title('Class Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main(annotations_path: str, imgs_path: str):
    # Folder path containing images, annotations, and classes.txt
    # folder_path = input("Enter the folder path: ")

    # Load classes and annotations
    # classes = load_classes(folder_path)
    classes = ('fish',)
    annotations = load_annotations(annotations_path)

    # Ask user for the bounding box colors for each class
    class_colors = {0: 'r'}
    # for i, class_name in enumerate(classes):
    #     color = input(f"Enter the color for {class_name} (e.g., 'r', 'g', 'b'): ")
    #     class_colors[i] = color

    # Plot class distribution
    # class_counts = count_class_instances(annotations, classes)
    # plot_class_distribution(class_counts)

    # Image navigation variables
    image_ids = sorted(annotations.keys())
    current_idx = 0

    # Create a figure and axes for display
    fig, ax = plt.subplots()

    def display_image_with_annotations():
        img_id = image_ids[current_idx]
        image = load_image(imgs_path, img_id)
        if image is not None:
            annotation_data = annotations.get(img_id, [])
            plot_annotations(ax, image, annotation_data, classes, class_colors)
        else:
            print(f"Image {img_id} not found.")

    # Display the first image
    display_image_with_annotations()

    # Function to scroll through images
    def on_key(event):
        nonlocal current_idx
        if event.key == 'right':
            current_idx = (current_idx + 1) % len(image_ids)
        elif event.key == 'left':
            current_idx = (current_idx - 1) % len(image_ids)
        display_image_with_annotations()

    # Connect the event handler for keypresses
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Display the figure
    plt.show()

if __name__ == "__main__":
    main('dataset/labels/train', 'dataset/images/train')