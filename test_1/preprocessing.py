import cv2 as cv
import numpy as np
import pickle


#Getting speed from the text file
speed = []
with open("data/train.txt") as f:
    speed = list(f)

speed = [x.strip() for x in speed]
iter_speed = iter(speed) #making the speed list an iterable

# retrieving all frames from video and saving them in all_frames
# all_frames[np.ndarry] = speed for that particular frame
cap = cv.VideoCapture("data/train.mp4")
all_frames = []
ret, frame1 = cap.read()
i = 0
while i < len(all_frames):
    temp = []
    temp.append(frame1)
    try:
        temp.append(float(next(iter_speed)))
    except:
        continue
    all_frames.append(temp)
    ret, frame1 = cap.read()
    i+=1

#applying Adaptive Guassian Thresholding to account for illumination changes
i = 0
while i < len(all_frames):
    img = cv.cvtColor(all_frames[i][0], cv.COLOR_BGR2GRAY) #convert to grayscale
    img = cv.medianBlur(img,5) #median blur

    #After applying adaptive gaussian thresholding, saving it to all_frames
    all_frames[i][0] = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                            cv.THRESH_BINARY, 11, 2)
    i += 1



#since video was analyze in pairs of successive frames
#80% of the intitial frames were training
#and 20% were used for validation
eighty_percent = int(0.8*len(all_frames))
training, validation = all_frames[:eighty_percent], all_frames[eighty_percent:]



#Dense Optical Flow based on the Gunner Farneback's algorithm
cap = cv.VideoCapture("data/train.mp4")
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv[...,1] = 255

i = 1
while i < len(training):
    next = training[i][0]

    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    cv.imshow('frame2',bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png',training[i][0])
        cv.imwrite('opticalhsv.png',bgr)
    prvs = next
    i += 1

cap.release()
cv.destroyAllWindows()
