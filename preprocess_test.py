import cv2


#Function to display the video of the current capture
def dispVid(cap):
    while True:
        ret, frame = cap.read()
        cv2.imshow("Name", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#Read training videp
cap = cv2.VideoCapture("data/test.mp4")

#Cropping video feed
x_start = 0
x_end = 640
y_start = 160
y_end = 370

fps = cap.get(cv2.CAP_PROP_FRAME_COUNT)
frame_count = 0
csv_write = []
while True:
    try:
        ret, frame = cap.read()
        frame = frame[y_start:y_end, x_start:x_end]
        frame_count+=1
        file_loc = "data/test_image/test_" + str(frame_count) + ".jpg"
        cv2.imwrite(file_loc, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        break
cap.release()
cv2.destroyAllWindows()
