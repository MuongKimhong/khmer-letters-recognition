from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np

from model.neural_net import KNeuralNet
from scripts import image_processing


print("[INFO] accessing images ....")
image_paths = list(paths.list_images("../khmer_letter_dataset/khmer_letter"))
dataset_loader = image_processing.DatasetLoader(image_paths, verbose=1)
(images, labels) = dataset_loader.load()
images = images.astype("float") / 255.0

print("[INFO] spliting images for train set and test set ....")
(trainX, testX, trainY, testY) = train_test_split(images, labels, test_size=0.1, random_state=42)

print("[INFO] converting labels to vectors ....")
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] loading model ....")
optimizer = SGD(learning_rate=0.005)
model = KNeuralNet.build(width=128, height=128, depth=3, classes=3)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

print("[INFO] training network ....")
train = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=500, verbose=1)

print("[INFO] saving model ....")
model.save("model/k_neural_net.hdf5")

print("[INFO] evaluating network ....")
predictions = model.predict(testX, batch_size=1)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=["kor", "khor", "kur"]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), train.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), train.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), train.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), train.history["val_accuracy"], label="val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch ")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()