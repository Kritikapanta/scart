#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from ultralytics import YOLO
import os

# Step 1: Define paths
dataset_yaml = r"C:\Datasets\datasets.yml"  # Path to your dataset.yaml file
model_path = "yolov8n.pt"      # Pre-trained YOLOv8 model (you can use yolov8s, yolov8m, etc.)

# Step 2: Load the YOLOv8 model
model = YOLO(model_path)

# Step 3: Train the model
results = model.train(
    data=dataset_yaml,  # Path to dataset.yaml
    epochs=28,          # Number of training epochs
    imgsz=640,          # Image size for training
    batch=16,           # Batch size
    name="keyboard_mouse_detection"  # Name of the training run
)

# Step 4: Evaluate the model
metrics = model.val()  # Evaluate on the validation set
print(f"mAP50-95: {metrics.box.map}")  # Print mAP score

# Step 5: Test the model on a sample image
sample_image_path = r"C:\Datasets\images"  # Path to a test image
results = model.predict(source=sample_image_path, save=True, conf=0.5)  # Run inference

# Step 6: Display results
for result in results:
    result.show()  # Show the image with detections
    result.save("C:\Datasets\detected")  # Save the image with detections

# Step 7: Export the model (optional)
model.export(format="onnx")  # Export to ONNX format for deployment


# In[ ]:




