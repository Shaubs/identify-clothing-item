import numpy as np
import cv2
from keras.preprocessing.image import img_to_array ,load_img
from keras.models import load_model

frame=cv2.imread(r"C:\\Users\\Subarna\\Documents\\haarTraining\\Haar Training\\training\\positive\\shirt\\17. 71-vjeR8-AL._UL1500_.jpg")
frame2=cv2.imread(r"C:\\Users\\Subarna\\Documents\\haarTraining\\Haar Training\\training\\positive\\shirt\\17. 71-vjeR8-AL._UL1500_.jpg")

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
imgray = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
ret4,thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


print ('all good')
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
cnt=contours[-1]

#loading model and specifications of classes
class_dictionary = np.load('class_pant_shirt.npy').item()
model=load_model('pant-shirt-model.h5')
model.compile(optimizer = 'adam',loss='categorical_crossentropy',
                              metrics = ['accuracy'])

#recievd box now preprocessing and predicting
x,y,w,h = cv2.boundingRect(cnt)
sub_rect=frame2[x:x+w,y:y+h]
sub_rect=img_to_array(sub_rect)
sub_rect=cv2.resize(sub_rect,(227,227))
#sub_rect=sub_rect.reshape(227,227,3)
sub_rect = np.expand_dims(sub_rect, axis=0)
sub_rect=sub_rect/225
result = model.predict(sub_rect)
pred_class=model.predict_classes(sub_rect)
inID = pred_class[0]
inv_map ={v:k for k,v in class_dictionary.items()}
label = inv_map[inID]
check=max(max(result))
if (check>0.7):
    print(label)
    cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
                    


frame2=cv2.resize(frame2,(200,200))
cv2.imshow('frame1',frame2)
cv2.waitKey(0)
cv2.destroyAllWindows()
