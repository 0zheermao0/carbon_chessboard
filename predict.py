# Author: Joey
# Email: zengjiayi666@gmail.com
# Date: :call strftime("%Y-%m-%d %H:%M")
# Description: 
# Version: 1.0
from ultralytics import YOLO
import cv2
from chessboard import get_chessboard_center
import numpy as np

# Load a model
model = YOLO('/mnt/d/chessboard/ultralytics/runs/detect/train/weights/last.pt')  # build from YAML and transfer weights

def get_chess_pos(img):
    results = model(img)
    types = []
    boxes = []
    for result in results:
        types.append(result.boxes.cls.cpu().tolist())
        boxes.append(result.boxes.xyxy.cpu().tolist())
    return types[0], boxes[0]

if __name__ == "__main__": 
    centers = get_chessboard_center(cv2.imread('/mnt/c/Users/Joey/Pictures/4.jpg'))

    img = cv2.imread('./datasets/storage/2023-05-20 16.25.33.jpg')

    pos = {}
    types, boxes = get_chess_pos(img)
    # 遍历每一个boxes，如果有一个centers在boxes里面，就把这个centers的index加入到pos里面
    # 其中pos的key是chess的index，value是chess的类型
    for i, box in enumerate(boxes):
        for j, center in enumerate(centers):
            if center[0] > box[0] and center[0] < box[2] and center[1] > box[1] and center[1] < box[3]:
                pos[j] = types[i]
                break
    # sort pos by key
    pos = dict(sorted(pos.items(), key=lambda item: item[0]))
    print(pos)

