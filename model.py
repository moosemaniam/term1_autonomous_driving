from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Convolution2D,Dropout,MaxPooling2D,Activation,Cropping2D

def nvidia_model(row,col,ch):
  model = Sequential()
  model.add(Lambda(lambda x:x/255.0 -
    0.5,input_shape=(row,col,ch),output_shape=(row,col,ch)))
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

  return model


