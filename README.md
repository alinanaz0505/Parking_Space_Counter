# ParkingSpacePicker

import cv2
import pickle
import cvzone  # Assuming this is an external library for drawing
import numpy as np

# Global variables
width, height = 107, 48
posList = []

# Function to save positions to a file using pickle
def save_positions():
    with open('CarParkPos', 'wb') as f:
        pickle.dump(posList, f)

# Function to load positions from a file
def load_positions():
    global posList
    try:
        with open('CarParkPos', 'rb') as f:
            posList = pickle.load(f)
    except:
        posList = []

# Function to handle mouse clicks
def mouseClick(events, x, y, flags, params):
    global posList
    if events == cv2.EVENT_LBUTTONDOWN:  # Left mouse button clicked
        posList.append((x, y))
    if events == cv2.EVENT_RBUTTONDOWN:  # Right mouse button clicked
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1 + width and y1 < y < y1 + height:
                posList.pop(i)
        save_positions()

# Main function to handle image annotation
def main():
    load_positions()
    while True:
        img = cv2.imread('carParkImg.png')
        for pos in posList:
            cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), (255, 0, 255), 2)
        cv2.imshow("Image", img)
        cv2.setMouseCallback("Image", mouseClick)
        key = cv2.waitKey(1)
        if key == ord('q'):  # Press 'q' to quit
            save_positions()
            break

# Check if script is running directly
if __name__ == "__main__":
    main()

# Main Parking Detection Program
cap = cv2.VideoCapture('carPark.mp4')  # Open video file

# Load parking space positions
with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

# Function to check parking space occupancy
def checkParkingSpace(imgPro):
    spaceCounter = 0
    for pos in posList:
        x, y = pos
        imgCrop = imgPro[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)
        if count < 900:
            color = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
        else:
            color = (0, 0, 255)
            thickness = 2
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1, thickness=2, offset=0, colorR=color)
    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3, thickness=5, offset=20, colorR=(0,200,0))

# Main loop to process video frames
while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)
    checkParkingSpace(imgDilate)
    cv2.imshow("Image", img)
    cv2.waitKey(10)
