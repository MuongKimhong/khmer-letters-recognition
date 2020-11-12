from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.utils import np_utils
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import cv2
import os

from model.neural_net import HandNeuralNet
from scripts import image_processing


print("[INFO] accessing images ....")
images = []
labels = []
image_paths = list(paths.list_images("../hand_dataset/hand"))

for image_path in sorted(image_paths):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image_processing.convert_to_array(image)
    images.append(image)
    label = image_path.split(os.path.sep)[-2]
    label = "left" if label == "left" else "right"
    labels.append(label)

images = np.array(images, dtype="float") / 255.0
labels = np.array(labels)

print("[INFO] converting labels to vectors ....")
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

print("[INFO] computing the class weights ....")
class_totals = labels.sum(axis=0)
class_weights = class_totals.max() / class_totals

print("[INFO] spliting images for train set and test set ....")
(trainX, testX, trainY, testY) = train_test_split(images, labels, test_size=0.1, stratify=labels, random_state=42)

print("[INFO] loading model ....")
model = HandNeuralNet.build(width=128, height=128, depth=1, classes=2)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print("[INFO] training network ....")
train = model.fit(trainX, trainY, validation_data=(testX, testY), class_weights=class_weights, batch_size=32, epochs=15, verbose=1)

print("[INFO] saving model ....")
model.save("model/hand_neural_net.hdf5")

print("[INFO] evaluating network ....")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), train.history["loss"], label="train_loss")
plt.plot(np.arange(0, 15), train.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 15), train.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 15), train.history["val_accuracy"], label="val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch ")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()