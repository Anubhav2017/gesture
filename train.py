import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
import cv2  
from sklearn.model_selection import train_test_split
# X=[]
# Y=[]

# foldernames=["palm","l","fist","fist_moved","thumb","index","ok","palm_moved","c","down"]
# imagepaths=[]
# for i in range(10):
#     for j in range(1,11):
#         for k in range(1,201):
#             filename="frame_{:02d}_{:02d}_{:04d}.png".format(i,j,k)
#             #print(filename)
#             cat=foldernames[j-1]
#             foldername="{:02d}_{}".format(j,cat)
#             path="leapGestRecog/{:02d}/{}/{}".format(i,foldername,filename)
#             imagepaths.append(path)

# for path in imagepaths:
#     img = cv2.imread(path) # Reads image and returns np.array
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts into the corret colorspace (GRAY)
#     img = cv2.resize(img, (320, 120)) # Reduce image size so training can be faster
#     X.append(img)
  
#     # Processing label in image path
#     category = path.split("/")[2]
#     label = int(category.split("_")[0][1]) # We need to convert 10_down to 00_down, or else it crashes
#     Y.append(label)
#     #print(label)

# # Turn X and y into np.array to speed up train_test_split
# X = np.array(X, dtype="uint8")
# X = X.reshape(len(imagepaths), 120, 320, 1) # Needed to reshape so CNN knows it's different images
# y = np.array(Y)

# np.save("X",X)
# np.save("y",y)

# print("Images loaded: ", len(X))
# print("Labels loaded: ", len(y))
ts=0.3
X=np.load("X.npy")
y=np.load("y.npy")

print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)

model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(120, 320, 1))) 
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
# Configures the model for training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Trains the model for a given number of epochs (iterations on a dataset) and validates it.
model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=2, validation_data=(X_test, y_test))

model.save("gesture_model")