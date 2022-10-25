from statistics import mode
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold

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
callback = EarlyStopping(monitor='val_accuracy', min_delta=0, 
patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=True)




#Model architecture
model = Sequential()

# Convolution - Transform image map in a smaller one
model.add(Convolution2D(32, (3, 3), 1, activation='relu', input_shape=(30, 30, 3)))
#For Spatial Invariance
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(128, (3, 3), 1,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(128, (3, 3), 1, activation='relu'))
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

array_rec = []
array_pre = []
array_f1 = []

n_for_train = int(len(X)*0.9)
n_of_our_validation = int(len(X)*0.1)


X_1, X_2 = X[:n_for_train], X[n_for_train:]
y_1, y_2 = y[:n_for_train], X[n_for_train:]

print(len(X_1))
print(len(X_2))
print(len(y_1))
print(len(y_2))

kfold = KFold(5)

for train_index, validate_index in kfold.split(X_1, y_1):
    X_train, X_validation = X_1[train_index], X_1[validate_index]
    y_train, y_validation = y_1[train_index], y_1[validate_index]
    
    #Fit model on training data
    hist = model.fit(datagen.flow(X_train, y_train, batch_size=64, 
    subset='training'),validation_data=datagen.flow(X_validation, y_validation, 
    batch_size=32, subset='validation'), epochs=10)

    y_predicted = model.predict(X_2)

    array_rec.append(precision_score(y_2, y_predicted))
    array_pre.append(recall_score(y_2, y_predicted))
    array_f1.append(f1_score(y_2, y_predicted))

    #predict 


#loss: 0.1284 - accuracy: 0.9491 - val_loss: 0.5026 - val_accuracy: 0.8597
#with less filters
#loss: 0.2678 - accuracy: 0.8887 - val_loss: 0.3479 - val_accuracy: 0.8501

#Visualize the models accuracy

print(f"Recall {array_rec}")
print(f"Precision {array_pre}")
print(f"F1-score {array_f1}")

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

