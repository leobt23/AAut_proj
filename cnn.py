import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Step 1 - Get data
X = np.load("Xtrain_Classification1.npy")
X_df = pd.DataFrame(X)
y = np.load("ytrain_Classification1.npy")
y_df = pd.DataFrame(y)

X = np.reshape(X, (8273, 30, 30, 3)) # 8k images 30x30 with 3 colours

# Scale data
#Change pixels to probabilities between 0 and 1
data_xtrain = X / 255


#Data augmentation
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1, 
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2)

datagen.fit(data_xtrain)



#One hot encoding class variables
data_ytrain = to_categorical(y)



#Stop training when accuracy has stopped improving
callback = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=3, 
                         verbose=0, mode='auto', baseline=None, 
                         restore_best_weights=True)



#Model architecture
model = Sequential()

model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(30, 30, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid')) #sigmoid is for binary data

#Compile model
model.compile(loss='binary_crossentropy',  #Even with one hot encoding?
              optimizer='adam',            #RMSprop? or adam?
              metrics=['accuracy'])

model.summary()

print("Here")


#Fit model on training data
hist = model.fit(datagen.flow(data_xtrain, data_ytrain, 
          batch_size=64, subset='training'),
                 validation_data=datagen.flow(data_xtrain, data_ytrain, 
        batch_size=32, subset='validation'), 
        epochs=60)

#val_loss: 0.2537 - val_accuracy: 0.9019



#Visualize the models accuracy
"""
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'lower right')
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'upper right')
plt.show()
"""