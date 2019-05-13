import cv2 as cv
from matplotlib import pyplot as plt

#Function to display the video of the current capture
def dispVid(cap):
    while True:
        ret, frame = cap.read()
        cv.imshow("Name", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

#Read training videp
cap = cv.VideoCapture("data/train.mp4")
#Cropping video feed
x_start = 0
x_end = 640
y_start = 160
y_end = 370
while True:
    ret, frame = cap.read()
    frame = frame[y_start:y_end, x_start:x_end]
    cv.imshow("Feed", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

