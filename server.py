# Author: Joey
# Email: zengjiayi666@gmail.com
# Date: :call strftime("%Y-%m-%d %H:%M")
# Description: 
# Version: 1.0

import socket
import cv2
from chessboard import get_chessboard_center
from predict import get_chess_boxes, get_chess_pos, detect_changed
import json
import os

config_path = './config.json'
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
        SERVER_IP = config['SERVER_IP']
        SERVER_PORT = config['SERVER_PORT']
        cam = config['CAM']
        WIDTH = config['WIDTH']
        HEIGHT = config['HEIGHT']
        STRLL_THRESHOLD = config['STRLL_THRESHOLD']
        MOVE_DETECT_THRESHOLD = config['MOVE_DETECT_THRESHOLD']

# 定义摄像头
cap = cv2.VideoCapture(cam)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)
# 摄像头是否打开，如果没打开则提示并终止程序
if not cap.isOpened():
    print('Error: Camera is not opened!')
    exit(-1)

# 获取第一帧
ret, frame = cap.read()
prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_pos = {}
# 获取棋盘格中心点
centers = get_chessboard_center(frame)
if centers != [] and len(centers) == 36:
    print(f"get centers {centers} successfully!")
else:
    print(f"no chessboard centers detected, initialize failed...")
    exit(1)
# centers = get_chessboard_center(cv2.imread('/mnt/c/Users/Joey/Pictures/4.jpg'))
frame_count = 0

if __name__ == "__main__":
    # 创建一个Socket对象
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # 绑定服务器IP地址和端口号
    server_socket.bind((SERVER_IP, SERVER_PORT))
    
    # 开始监听客户端连接请求
    print("waiting conn")
    server_socket.listen()
    client_socket, address = server_socket.accept()
    
    welcome_msg = 'connect success!'
    # client_socket.send(welcome_msg.encode())
    
    # 不断检测画面变动并向客户端发送消息
    while True:
        # 获取当前帧
        ret, frame = cap.read()
        curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv 
    
        # 比较当前帧和前一帧
        diff = cv2.absdiff(curr_frame, prev_frame)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.resize(thresh, (480, 480))
        # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        if cv2.countNonZero(thresh) > MOVE_DETECT_THRESHOLD:
            print(f"camera detected movement, recording changed...\n")
        else:
            # changed_pos 不为空时发送消息
            frame_count += 1
            types, boxes = get_chess_boxes(frame)
            curr_pos = get_chess_pos(centers, types, boxes)
            changed_pos = detect_changed(prev_pos, curr_pos)
            if changed_pos != {} and changed_pos != None and frame_count >= STILL_THRESHOLD:
                try: 
                    print(str(changed_pos).encode())
                    client_socket.send(str(changed_pos).encode())
                except ConnectionResetError as cre:
                    # 重新等待客户端连接
                    print(f"client disconnected, waiting for new connection...\n")
                    client_socket, address = server_socket.accept()
                    client_socket.send(str(changed_pos).encode())
                except Exception as e:
                    print(e)
                    continue
                frame_count = 0
            print(f"camera detected no movement, preparing to send msg...\n")
            prev_pos = curr_pos
    
        # 更新前一帧
        prev_frame = curr_frame
    
    # 关闭Socket连接
    cap.release()
    client_socket.close()
    server_socket.close()

