import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import cv2
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.applications import VGG16
from sklearn.model_selection import train_test_split

np.random.seed(1)

train_images = []
train_labels = []
shape = (200, 200)
train_path = './Pictures/train'

for filename in os.listdir('./Pictures/train'):
    if filename.split('.')[1] == 'jpeg' or filename.split('.')[1] == 'jpg':
        img = cv2.imread(os.path.join(train_path, filename))

        train_labels.append(filename.split('_')[0])

        img = cv2.resize(img, shape)

        train_images.append(img)

train_labels = pd.get_dummies(train_labels).values

train_images = np.array(train_images)

x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, random_state=1)


test_images = []
test_labels = []
shape = (200, 200)
test_path = './Pictures/test'

for filename in os.listdir('./Pictures/test'):
    if filename.split('.')[1] == 'jpeg' or filename.split('.')[1] == 'jpg':
        img = cv2.imread(os.path.join(test_path, filename))

        test_labels.append(filename.split('_')[0])
        img = cv2.resize(img, shape)

        test_images.append(img)

test_images = np.array(test_images)


plt.imshow(train_images[0])

plt.imshow(train_images[4])


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(200, 200, 3))


for layer in base_model.layers:
    layer.trainable = False


x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(train_labels.shape[1], activation='softmax')(x)


model = Model(inputs=base_model.input, outputs=predictions)


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model.fit(x_train, y_train, epochs=50, batch_size=50, validation_data=(x_val, y_val))


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


model.save('fruit_prediction_transfer.h5')


evaluate = model.evaluate(x_val, y_val)
print(evaluate)

checkImage = test_images[20]
checklabel = test_labels[20]

predict = model.predict(np.array([checkImage]))

output = {0: 'apple', 1: 'banana', 2: 'mixed', 3: 'orange'}

print("Actual :- ", checklabel)
print("Predicted :- ", output[np.argmax(predict)])
