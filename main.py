from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path

def train_model():
  
    model = YOLO('yolov8n.pt')  

    results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=4, 
    augment=True,  
    fliplr=0.5,    
    scale=0.5,     
    translate=0.1,
    degrees=10,    
    mosaic=1.0,    
    mixup=0.0      
)

    return results

def draw_bounding_boxes(image, detections, class_names):
    """
    Draw bounding boxes on the image for detected objects.
    :param image: Input image (numpy array).
    :param detections: YOLO detection results.
    :param class_names: List of class names.
    :return: Image with bounding boxes drawn.
    """
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
    
        cv2.rectangle(
            image,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),  
            2  
        )
        label = f"{class_names[int(cls)]}: {conf:.2f}"
        cv2.putText(
            image,
            label,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    return image

def get_latest_weights():
    """
    Find the latest trained YOLO weights (best.pt) dynamically.
    :return: Path to the best.pt file.
    """
    runs_dir = Path('runs/detect')
    if not runs_dir.exists():
        raise FileNotFoundError("No training results found in 'runs/detect/' directory!")

    experiment_dirs = sorted(runs_dir.glob('*'), key=os.path.getmtime, reverse=True)
    if not experiment_dirs:
        raise FileNotFoundError("No experiment directories found in 'runs/detect/'!")

    latest_experiment = experiment_dirs[0]
    best_weights = latest_experiment / 'weights/best.pt'

    if not best_weights.exists():
        raise FileNotFoundError(f"Best weights not found at {best_weights}!")
    return best_weights

def predict_on_image_and_save(model_path, image_path, output_path, conf_threshold=0.25):
    """
    Run inference on an image and save the output with bounding boxes.
    :param model_path: Path to the trained YOLO model weights.
    :param image_path: Path to the input image.
    :param output_path: Path to save the output image.
    :param conf_threshold: Confidence threshold for predictions.
    """
    model = YOLO(model_path)

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")

    results = model.predict(source=img, conf=conf_threshold, verbose=False)

    import yaml
    with open('data.yaml', 'r') as file:
        class_names = yaml.safe_load(file)['names']

    detections = results[0].boxes.data.cpu().numpy()

    output_img = draw_bounding_boxes(img.copy(), detections, class_names)

    cv2.imwrite(output_path, output_img)
    print(f"Output saved at {output_path}")

def main():
    Path('predictions/tree_detection').mkdir(parents=True, exist_ok=True)

    print("Starting training...")
    train_results = train_model()
    print("Training completed!")

    try:
        best_model_path = get_latest_weights()
        print(f"Using trained model: {best_model_path}")

        input_image = 'input.jpg'
        output_image = 'output.png'
        predict_on_image_and_save(
            model_path=best_model_path,
            image_path=input_image,
            output_path=output_image
        )
        print(f"Inference completed! Output saved as {output_image}.")
    except FileNotFoundError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
