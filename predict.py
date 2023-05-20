# Author: Joey
# Email: zengjiayi666@gmail.com
# Date: :call strftime("%Y-%m-%d %H:%M")
# Description: 
# Version: 1.0
from ultralytics import YOLO
import cv2
from chessboard import get_chessboard_center

# Load a model
model = YOLO('/mnt/d/chessboard/ultralytics/runs/detect/train/weights/last.pt')  # build from YAML and transfer weights

if __name__ == "__main__": 
    centers = get_chessboard_center(cv2.imread('/mnt/c/Users/Joey/Pictures/4.jpg'))

    img = cv2.imread('./datasets/chess/19776099095189541.jpg')
    # img = cv2.imread('./datasets/test/images/151258738916556852_jpg.rf.3763b4a5255b13cc091f16652ca9b2f7.jpg')
    cv2.imshow('img', img)
    model.predict(img, save=True)
    results = model(img)
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        probs = result.probs  # Class probabilities for classification outputs
        print(f"boxes: {boxes}, masks: {masks}, probs: {probs}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

