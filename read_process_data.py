import csv
import cv2
import numpy as np
limit=1000
lines=[]
with open('data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile,delimiter=',')
  for row in reader:
    lines.append(row)

images=[]
measurements=[]
index=-1
for line in lines:
  #Ignore first row which contains labels
  if(index==limit):
    continue
  if(index == -1):
    index=0
    continue


  source_path=line[0]
  filename = source_path.split('/')[-1]
  current_path = 'data/IMG/' + filename
  image = cv2.imread(current_path)
  measurement = float(line[3])
#Only take non zero index values
  if(measurement!=0.0):
   print(measurement)
   image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB)
   images.append(image)
   measurements.append(measurement)
   index+=1

X_data=np.array(images)
y_data=np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Convolution2D,Dropout,MaxPooling2D,Activation

model = Sequential()
model.add(Lambda(lambda x:x/255.0 - 0.5,input_shape=(160,320,3)))
model.add(Convolution2D(32, 3, 3))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Convolution2D(32, 3, 3))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_data,y_data,validation_split=0.2,shuffle=True,nb_epoch=4)
model.save("model.h5")

