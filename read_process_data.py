import csv
import cv2
import numpy as np
lines=[]
correction_factor=0.3
with open('data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile,delimiter=',')
  for row in reader:
    lines.append(row)

images=[]
measurements=[]
index=-1
for line in lines:
  #Ignore first row which contains labels
  if(index>24000):
    break
  if(index == -1):
    index=0
    continue
  source_path=line[0]
  filename = source_path.split('/')[-1]
  current_path = 'data/IMG/' + filename
  image = cv2.imread(current_path)
  measurement = float(line[3].replace(",","."))
  image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB)
  images.append(image)
  measurements.append(measurement)
  #Also add in the flipped imaged
  images.append(np.fliplr(image))
  measurements.append(-measurement)

  #Left images
  source_path=line[1]
  filename = source_path.split('/')[-1]
  current_path = 'data/IMG/' + filename
  image = cv2.imread(current_path)
  measurement = float(line[3].replace(",","."))
  image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB)
  images.append(image)
  measurement=measurement+correction_factor
  measurements.append(measurement)
  #Also add in the flipped imaged
  images.append(np.fliplr(image))
  measurements.append(-measurement)

  #Right images
  source_path=line[2]
  filename = source_path.split('/')[-1]
  current_path = 'data/IMG/' + filename
  image = cv2.imread(current_path)
  measurement = float(line[3].replace(",","."))
  image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB)
  images.append(image)
  measurement=measurement-correction_factor
  measurements.append(measurement)
  #Also add in the flipped imaged
  images.append(np.fliplr(image))
  measurements.append(-measurement)
  index+=6

X_data=np.array(images)
y_data=np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Convolution2D,Dropout,MaxPooling2D,Activation,Cropping2D

model = Sequential()
model.add(Lambda(lambda x:x/255.0 - 0.5,input_shape=(160,320,3)))
#model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(24, 5,
  5,subsample=(2,2),activation="relu",border_mode="valid"))

model.add(Convolution2D(36, 5,
5,subsample=(2,2),activation="relu",border_mode="valid"))

model.add(Convolution2D(48, 5,
5,subsample=(2,2),activation="relu",border_mode="valid"))

model.add(Convolution2D(64, 3, 3,activation="relu",border_mode="valid"))

model.add(Convolution2D(64, 3, 3,activation="relu",border_mode="valid"))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_data,y_data,validation_split=0.2,shuffle=True,nb_epoch=2)
model.save("model.h5")

