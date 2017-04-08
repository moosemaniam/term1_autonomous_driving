import csv
import cv2
import numpy as np

lines=[]
with open('data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile,delimiter=',')
  for row in reader:
    lines.append(row)

images=[]
measurements=[]
for line in lines:
  source_path=line[0]
  filename = source_path.split('/')[-1]
  current_path = 'data/IMG/' + filename
  image = cv2.imread(current_path)
  images.append(image)
  measurement = float(line[3])
  measurements.append(measurement)

X_data=np.array(images)
y_data=np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten,Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_data,y_data,validation_split=0.2,shuffle=True,nb_epoch=7)
model.save("model_basic")

