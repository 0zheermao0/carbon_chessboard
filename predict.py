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
model = YOLO('./models/chess.pt')  # build from YAML and transfer weights

def get_chess_boxes(img):
    results = model(img)
    types = []
    boxes = []
    for result in results:
        types.append(result.boxes.cls.cpu().tolist())
        boxes.append(result.boxes.xyxy.cpu().tolist())
    return [int(i) for i in types[0]], boxes[0]

def get_chess_pos(centers, types, boxes):
    # 遍历每一个boxes，如果有一个centers在boxes里面，就把这个centers的index加入到pos里面
    # 其中pos的key是chess的index，value是chess的类型
    pos = {}
    for i, box in enumerate(boxes):
        for j, center in enumerate(centers):
            if center[0] > box[0] and center[0] < box[2] and center[1] > box[1] and center[1] < box[3]:
                pos[j] = types[i]
                break
    # sort pos by key
    pos = dict(sorted(pos.items(), key=lambda item: item[0]))
    return pos

def detect_changed(prev_pos: dict, curr_pos: dict) -> list:
    # compare prev_pos and curr_pos
    # if new chess added, return {type: 1, index: {changed chess board index}, build: {changed chess type}}
    # if chess removed, return {type: 0, index: {changed chess board index}, build: {changed chess type}}
    results = []
    for key in prev_pos.keys():
        if key not in curr_pos.keys():
            results.append({'type': 0, 'index': key, 'build': prev_pos[key]})
    for key in curr_pos.keys():
        if key not in prev_pos.keys():
            results.append({'type': 1, 'index': key, 'build': curr_pos[key]})

    return results

# if __name__ == "__main__": 
#     centers = get_chessboard_center(cv2.imread('/mnt/c/Users/Joey/Pictures/4.jpg'))
# 
#     img = cv2.imread('./datasets/storage/2023-05-20 16.25.33.jpg')
# 
#     types, boxes = get_chess_boxes(img)
#     pos = get_chess_pos(centers, types, boxes)
#     print(pos)

