from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
import cv2
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import load_model
from keras.preprocessing import image




batch_size=10
num_classes=2

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(227,227,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'path to training data',  
        target_size=(227,227),
        batch_size=batch_size,
        class_mode='categorical') 
num_classes = len(train_generator.class_indices)
np.save('class_pant_shirt.npy', train_generator.class_indices)
train_labels = train_generator.classes
train_labels = to_categorical(train_labels, num_classes=num_classes)

print("------------printing num_classes:")
print(num_classes)

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'path to validation data',
        target_size=(227,227),
        batch_size=batch_size,
        class_mode='categorical')

validation_labels = validation_generator.classes
validation_labels = to_categorical(validation_labels,num_classes=num_classes)
model.fit_generator(
        train_generator,
        steps_per_epoch=40,
        epochs=25,
        validation_data=validation_generator,
        validation_steps=73)
#model.save_weights('first_try.h5')
model.save('pant-shirt-model.h5')
print("----------------training,validating done,save it------------------now have to test")

#testing

def predict():
    # load the class_indices saved in the earlier step
    class_dictionary = np.load('class_pant_shirt.npy').item()

    #model
    model=load_model('pant-shirt-model.h5')
    model.compile(optimizer = 'adam',
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])


    # add the path to your test image below
    image_path = 'path to image that has to undergo prediction'

    orig = cv2.imread(image_path)
    orig = cv2.resize(orig,(227,227))
    

    print("[INFO] loading and preprocessing image...")
    #image = load_img(image_path, target_size=(227, 227))
    image = img_to_array(image_path)
    image=cv2.resize(image,(227,227))
    #image = cv2.resize(image,(227,227))
    image = np.expand_dims(image, axis=0)
    # important! otherwise the predictions will be '0'
    image = image / 255


   
    prediction=model.predict(image)

    #model.load_weights('first_try.h5')
    class_predicted=model.predict_classes(image)
    
    probabilities=model.predict_proba(image)
   

    

    inID = class_predicted[0]
    inv_map ={v:k for k,v in class_dictionary.items()}
    label = inv_map[inID]

    print("Image ID:{},Label:{}".format(inID,label))
    cv2.putText(orig,"Predicted:{}".format(label),(10,30),cv2.FONT_HERSHEY_PLAIN,
                1.5,(43,99,255),2)

    cv2.imshow("Classification",orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

predict()
cv2.destroyAllWindows

