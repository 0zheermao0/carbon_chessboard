"""
    Author: Joey
    Date: 2023/05/09 22:22:16
    Email: zengjiayi666@gmail.com
    Description: 识别棋盘格
"""
import cv2
import numpy as np
import argparse

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--image', type=str, default='/mnt/c/Users/Joey/Pictures/20.png', help='image path')
    args = parse.parse_args()
    # Load image
    img = cv2.imread(args.image)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    # close操作去除杂点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cv2.imshow('thresh', thresh)
    
    # 查找轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
    # 去掉最大的轮廓(大棋盘外围轮廓)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:]
    # 找到能近似矩形且面积不太小的轮廓
    approx_contours = []
    for contour in contours:
        epsilon = 0.15 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(approx)
        if len(approx) == 4 and area > 1500:
            approx_contours.append(approx)
    
    # 绘制这些矩形的中心点
    centers = []
    for contour in approx_contours:
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        centers.append((cx, cy))
        # cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
    
    # 去除相隔距离过近的点
    centers.sort(key=lambda x: x[0])
    centers.sort(key=lambda x: x[1])
    centers = np.array(centers)
    centers = np.unique(centers, axis=0)
    for center in centers:
        cv2.circle(img, tuple(center), 5, (255, 0, 0), -1)
    
    # 打印所有点的坐标
    i = 0
    for center in centers:
        print(f"center {i}: {center}")
        i += 1
    
    # 绘制轮廓
    cv2.drawContours(img, approx_contours, -1, (0, 0, 255), 2)
    print(f"centers amount: {len(centers)}, approx_contours amount: {len(approx_contours)}")
    
    cv2.imshow('Chessboard', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    
