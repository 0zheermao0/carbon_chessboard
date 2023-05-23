# Author: Joey
# Email: zengjiayi666@gmail.com
# Date: :call strftime("%Y-%m-%d %H:%M")
# Description: 
# Version: 1.0
#
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s.yaml')  # build a new model from YAML
model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8s.yaml').load('/mnt/d/chessboard/yolov8s.pt')  # build from YAML and transfer weights

# Train the model
if __name__ == "__main__": 
    model.train(data='./datasets/data.yaml', epochs=300, imgsz=640, patience=100, batch=16)

