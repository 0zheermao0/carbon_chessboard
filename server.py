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

# 定义摄像头
cap = cv2.VideoCapture(cam)
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
# centers = get_chessboard_center(cv2.imread('/mnt/c/Users/Joey/Pictures/4.jpg'))

if __name__ == "__main__":
    # 创建一个Socket对象
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # 绑定服务器IP地址和端口号
    server_socket.bind((SERVER_IP, SERVER_PORT))
    
    # 开始监听客户端连接请求
    server_socket.listen()
    client_socket, address = server_socket.accept()
    
    welcome_msg = 'connect success!'
    client_socket.send(welcome_msg.encode())
    
    # 不断检测画面变动并向客户端发送消息
    while True:
        # 获取当前帧
        ret, frame = cap.read()
        curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # 比较当前帧和前一帧
        diff = cv2.absdiff(curr_frame, prev_frame)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        # 如果画面有变动，则向客户端发送消息
        if len(contours) > 0:
            # frame = cv2.imread('./datasets/storage/2023-05-20 16.25.33.jpg')
            print(f"camera detected movement, sending message...\n")
            types, boxes = get_chess_boxes(frame)
            curr_pos = get_chess_pos(centers, types, boxes)
            changed_pos = detect_changed(prev_pos, curr_pos)
            if changed_pos != []:
                try: 
                    client_socket.send(str(changed_pos).encode())
                except Exception as e:
                    # 重新等待客户端连接
                    print(f"client disconnected, waiting for new connection...\n")
                    client_socket, address = server_socket.accept()
                    client_socket.send(str(changed_pos).encode())
    
        # 更新前一帧
        prev_frame = curr_frame
        prev_pos = curr_pos
    
    # 关闭Socket连接
    cap.release()
    client_socket.close()
    server_socket.close()
