import numpy as np
import cv2
frame=cv2.imread(r"C:\Users\Subarna\Documents\haarTraining\Haar Training\training\positive\rawdata\a21.bmp")
print type(frame)

#cv2.imshow('broooo',frame)
blur = cv2.bilateralFilter(frame,11,75,75)
frame = cv2.medianBlur(blur,19)
z=frame.reshape((-1,3))
#convert to np.float32
z=np.float32(z)
#define criteria, no of klusters and apply kmeans()
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
k=2
ret,label,center=cv2.kmeans(z,k,None,criteria,1,cv2.KMEANS_RANDOM_CENTERS)
#now back to unit8, original image
center=np.uint8(center)
res=center[label.flatten()]
res2=res.reshape((frame.shape))
cv2.imshow('es2',res2)
imgray = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
ret4,thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

cv2.imshow('thresh',thresh)
print ('all good')
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
cnt=contours[-1]
#image = cv2.drawContours(res2,contours,1, (0,255,0), 3)
cv2.drawContours(res2,(cnt),1, (0,255,0), 3)

img = cv2.drawContours(frame, [cnt], 0, (0,255,0), 3)
x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(res2,(x,y),(x+w,y+h),(0,255,0),2)

rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(res2,[box],0,(0,0,255),2)


cv2.imshow('frame1',res2)
cv2.imshow('frame',frame)

    
x,y,w,h = cv2.boundingRect(cnt)
aspect_ratio = float(w)/h
print (aspect_ratio)
area = cv2.contourArea(cnt)
x,y,w,h = cv2.boundingRect(cnt)
rect_area = w*h
extent = float(area)/rect_area
print (extent)
area = cv2.contourArea(cnt)
hull = cv2.convexHull(cnt)
hull_area = cv2.contourArea(hull)
solidity = float(area)/hull_area
print (solidity)

area = cv2.contourArea(cnt)
equi_diameter = np.sqrt(4*area/np.pi)
print(equi_diameter)


cv2.waitKey(0)
cv2.destroyAllWindows()
    
