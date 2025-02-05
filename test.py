from ultralytics import YOLO
import os

# Load the trained model
trained_model_path = r"C:\Users\Acer\runs\detect\keyboard_mouse_detection8\weights\best.pt"  # Adjust path if needed
model = YOLO(trained_model_path)

# Path to test images (folder containing images, not a single image file)
test_images_path = r"C:\Datasets\testing" # Folder containing test images
output_path = r"C:\Datasets\detected"  # Folder to save detected images

# Create output folder if not exists
os.makedirs(output_path, exist_ok=True)

# Run inference on test images
for image_name in os.listdir(test_images_path):
    image_path = os.path.join(test_images_path, image_name)
    
    # Check if it's an image file (optional)
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Perform detection
        results = model.predict(source=image_path, conf=0.5, save=True)
        
        # Display results
        for result in results:
            result.show()  # Show image with detections
            result.save(output_path)  # Save detected image

print("Testing completed. Detected images saved to:", output_path)
