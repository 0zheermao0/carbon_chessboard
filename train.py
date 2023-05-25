# Author: Joey
# Email: zengjiayi666@gmail.com
# Date: :call strftime("%Y-%m-%d %H:%M")
# Description: 
# Version: 1.0
#
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n.yaml').load('/mnt/d/chessboard/ultralytics/runs/detect/train2/weights/best.pt')  # build from YAML and transfer weights

# Train the model
if __name__ == "__main__": 
    model.train(data='./datasets/data.yaml', epochs=300, imgsz=640, patience=150, batch=16)

