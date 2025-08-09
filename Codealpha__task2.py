import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import os
import cv2

# COCO class labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
    'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
    'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Device setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load pretrained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()

# List of COCO classes considered as jars (mapping bottle, cup, wine glass to "jar")
JAR_CLASSES = {'bottle', 'cup', 'wine glass'}

def detect_on_image(image, threshold=0.7):
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(image_tensor)

    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in predictions[0]['labels'].cpu()]
    pred_boxes = predictions[0]['boxes'].cpu()
    pred_scores = predictions[0]['scores'].cpu()

    filtered = []
    for box, label, score in zip(pred_boxes, pred_classes, pred_scores):
        if score > threshold:
            # If label is in jar classes, rename to 'jar'
            display_label = 'jar' if label in JAR_CLASSES else label
            filtered.append((box, display_label, score))

    return filtered

def draw_boxes_pil(image, detections):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1, figsize=(12,9))
    ax.imshow(image)

    for box, label, score in detections:
        xmin, ymin, xmax, ymax = box.tolist()
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            xmin, ymin - 5, f'{label}: {score:.2f}',
            color='red', fontsize=12, backgroundcolor='white'
        )

    ax.text(
        0.98, 0.02, f'Total Objects: {len(detections)}',
        verticalalignment='top', horizontalalignment='right',
        transform=ax.transAxes, fontsize=14, color='blue',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='blue')
    )

    plt.axis('off')
    plt.tight_layout()
    plt.show()

def run_image_detection(image_path, threshold=0.7):
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
        detections = detect_on_image(image, threshold)
        draw_boxes_pil(image, detections)
    else:
        print(f"Image not found or path not provided. Switching to webcam...")
        run_webcam_detection(threshold)

def run_webcam_detection(threshold=0.7):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit webcam detection")

    frame_count = 0
    detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_count += 1

        # Run detection every 5 frames to reduce lag
        if frame_count % 5 == 0:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detections = detect_on_image(pil_image, threshold)

        # Draw boxes from last detections
        for box, label, score in detections:
            xmin, ymin, xmax, ymax = box.int().tolist()
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.putText(frame, f"{label}: {score:.2f}", (xmin, max(ymin - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(frame, f"Total Objects: {len(detections)}", (frame.shape[1] - 300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Webcam Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting webcam detection.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Put your image path here or None to directly open webcam
    image_path = None  # e.g. "your_image.jpg" or None

    run_image_detection(image_path)
