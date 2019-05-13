import cv2 as cv
from matplotlib import pyplot as plt
import pickle

#Function to display the video of the current capture
def dispVid(cap):
    while True:
        ret, frame = cap.read()
        cv.imshow("Name", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

#Read training videp
cap = cv.VideoCapture("data/train.mp4")
speed = []
with open("data/train.txt") as f:
    for el in f:
        speed.append(float(el.rstrip('\n')))
iter_speed = iter(speed)
#Cropping video feed
x_start = 0
x_end = 640
y_start = 160
y_end = 370

fps = cap.get(cv.CAP_PROP_FRAME_COUNT)
frame_count = 0
csv_write = []
while True:
    try:
        ret, frame = cap.read()
        frame = frame[y_start:y_end, x_start:x_end]
        frame_count+=1
        file_loc = "data/train_image/train_" + str(frame_count) + ".jpg"
        cv.imwrite(file_loc, frame)
        string = file_loc + "," + str(frame_count) + "," + str(next(iter_speed))
        csv_write.append(string)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        break
cap.release()
cv.destroyAllWindows()


output = open('data.pkl', 'wb')
pickle.dump(csv_write, output)
