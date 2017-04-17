import csv
import cv2
import numpy as np
import sklearn
from network import nvidia_model

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Convolution2D,Dropout,MaxPooling2D,Activation,Cropping2D

BATCH_SIZE=72

def generator(samples, batch_size=BATCH_SIZE):
    num_samples=len(samples)
    while 1: # Loop forever so the generator never terminates
        images = []
        angles = []
        batch_size_filled=0
        correction=0.25
        #Loop till we have batch size
        while (batch_size_filled < batch_size):
          #pick a random sample
          line = samples[np.random.randint(num_samples)]

##########Center image
          sample= line[0]
          name = 'data/IMG/'+ sample.split('/')[-1]
          center_image = cv2.imread(name)
          #Change color to RGB
          center_image = cv2.cvtColor( center_image, cv2.COLOR_BGR2RGB)
          center_angle= float(line[3].replace(",","."))
          if center_image is None or center_angle == 0.0:
            continue
          images.append(center_image)
          angles.append(center_angle)
          batch_size_filled +=1

          #Flipped Image
          images.append(np.fliplr(center_image))
          angles.append(-center_angle)
          batch_size_filled +=1

##########Load left image
          sample= line[1]
          name = 'data/IMG/'+ sample.split('/')[-1]
          left_image = cv2.imread(name)
          #Change color to RGB
          left_image = cv2.cvtColor( left_image, cv2.COLOR_BGR2RGB)
          left_angle= float(line[3].replace(",","."))
          #Since the camera is to the left,
          #we want the car to run right
          left_angle = left_angle + correction
          if left_image is None or left_angle == 0.0:
            continue
          images.append(left_image)
          angles.append(left_angle)
          batch_size_filled +=1

          #Flipped Image
          images.append(np.fliplr(left_image))
          angles.append(-left_angle)
          batch_size_filled +=1

##########Load right image
          sample= line[2]
          name = 'data/IMG/'+ sample.split('/')[-1]
          right_image = cv2.imread(name)
          #Change color to RGB
          right_image = cv2.cvtColor( right_image, cv2.COLOR_BGR2RGB)
          right_angle= float(line[3].replace(",","."))
          #Since the camera is to the right,
          #we want the car to run right
          right_angle = right_angle - correction
          if right_image is None or right_angle == 0.0:
            continue
          images.append(right_image)
          angles.append(right_angle)
          batch_size_filled +=1

          #Flipped Image
          images.append(np.fliplr(right_image))
          angles.append(-right_angle)
          batch_size_filled +=1


        yield np.array(images),np.array(angles)



#Read string data from csv file
samples=[]
with open('data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile,delimiter=',')
  for row in reader:
    samples.append(row)

#Ignore first row
samples=samples[1:]
#samples now has the entire string data from the CSV file

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#Data augmentation  increases the amount of data
train_sample_len = 6 * len(train_samples)
validation_sample_len = 6 * len(validation_samples)

## compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)


ch, row, col = 3, 160, 320  # Trimmed image format

model = nvidia_model(row,col,ch)

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator,samples_per_epoch=train_sample_len,validation_data=validation_generator,nb_val_samples=validation_sample_len,nb_epoch=3)
#model.save("model.h5")
