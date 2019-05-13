#code taken from Simple ROI - Medium
#Example code for cropping for the car train data
import cv2 as cv
from matplotlib import pyplot as plt

def sketch_transform(image):
    image_grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_grayscale_blurred = cv.GaussianBlur(image_grayscale, (7,7),0)
    image_canny = cv.Canny(image_grayscale_blurred, 10, 80)
    _, mask = image_canny_inverted = cv.threshold(image_canny, 30, 255, cv.THRESH_BINARY_INV)

    return mask

cam_capture = cv.VideoCapture(0)
cv.destroyAllWindows()

upper_left = (50,50)
bottom_right = (300,300)

while True:
    _, image_frame = cam_capture.read()

    #Rectangle marker
    r = cv.rectangle(image_frame, upper_left, bottom_right, (100,50,200), 5)
    rect_img = image_frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]

    sketcher_rect = rect_img
    sketcher_rect = sketch_transform(sketcher_rect)

    #Conversion for 3 channels to put back on original image (streaming)
    sketcher_rect_rgb = cv.cvtColor(sketcher_rect, cv.COLOR_GRAY2RGB)

    #Replacing the sketched image on ROI
    image_frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]] = sketcher_rect_rgb

    cv.imshow("Sketcher ROI", image_frame)
    if cv.waitKey(1) == 13:
        break
cam_capture.release()
cv.destroyAllWindows()