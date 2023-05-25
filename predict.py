# Email: zengjiayi666@gmail.com
# Date: :call strftime("%Y-%m-%d %H:%M")
# Description: 
# Version: 1.0
from ultralytics import YOLO
import cv2
from chessboard import get_chessboard_center
import numpy as np
import json
import os

# load config
config_path = './config.json'
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
        MODEL_PATH = config['MODEL_PATH']

# Load a model
model = YOLO(MODEL_PATH)  # build from YAML and transfer weights

def calculate_distance(point1, point2):
    # 计算两点之间的欧几里德距离
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def determine_chessboard_position(chess_piece_box, centers):
    # 计算棋子中心点坐标
    chess_piece_center = ((chess_piece_box[0] + chess_piece_box[2]) // 2, (chess_piece_box[1] + chess_piece_box[3]) // 2)

    # 初始化最小距离和落点位置
    min_distance = float('inf')
    target_position = None

    # 遍历棋盘格子中心点
    for i, center in enumerate(centers):
        # 计算棋子中心点与格子中心点之间的距离
        distance = calculate_distance(chess_piece_center, center)

        # 更新最小距离和落点位置
        if distance < min_distance:
            min_distance = distance
            target_position = i

    return target_position

def get_chess_boxes(img):
    results = model(img)
    types = []
    boxes = []
    for result in results:
        types.append(result.boxes.cls.cpu().tolist())
        boxes.append(result.boxes.xyxy.cpu().tolist())
    types_map = {0: 7, 2: 12, 3: 10, 4: 14, 5: 11, 6: 6, 7: 8, 8: 4, 9: 5, 10: 13, 11: 9, 12: 15}
    for i, t in enumerate(types[0]):
        if t == 1:
            types[0].pop(i)
            boxes[0].pop(i)
        else:
            types[0][i] = types_map[t]

    return [int(i) for i in types[0]], boxes[0]

def get_chess_pos(centers, types, boxes):
    # 其中pos的key是chess的index，value是chess的类型
    # 如果centers中的点在boxes中的某个box内，则认为该棋子在该中心点对应的格子内
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
    # if change number > 1, return None
    if len(prev_pos) - len(curr_pos) > 1 or len(curr_pos) - len(prev_pos) > 1:
        print(f"detect changed function detect more than one change, return None")
        return None

    # remove
    for key in prev_pos.keys():
        if key not in curr_pos.keys():
            return {'type': 0, 'index': key, 'build': prev_pos[key]}
    # add
    for key in curr_pos.keys():
        if key not in prev_pos.keys():
            return {'type': 1, 'index': key, 'build': curr_pos[key]}

    print(f"detect changed function detect no change, return None")    
    return None


# if __name__ == "__main__": 
#     centers = get_chessboard_center(cv2.imread('/mnt/c/Users/Joey/Pictures/4.jpg'))
# 
#     img = cv2.imread('./datasets/storage/2023-05-20 16.25.33.jpg')
# 
#     types, boxes = get_chess_boxes(img)
#     pos = get_chess_pos(centers, types, boxes)
#     print(pos)


