import numpy as np
import cv2

#Import and process image to obtain contours
im = cv2.imread('real_apple.jpg')
hsv_img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
imgray = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)
_, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# Find the index of the largest contour
areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cnt=contours[max_index]

#Overlay bounding box and show in new window
x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)        
cv2.imshow("Show",im)
cv2.imshow("Show gray scale", imgray)
cv2.imshow("Show HSV", hsv_img)
cv2.imshow("Show thresh", thresh)
cv2.waitKey()
cv2.destroyAllWindows()