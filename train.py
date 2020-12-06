from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np

from model.neural_net import KNeuralNet
from scripts import image_processing


def train(dataset_shape, dataste_path, test_size, img_width, img_height, img_depth, classes, batch_size, epochs, target_names):
    print("[INFO] accessing images ....")
    image_paths = list(paths.list_images(dataset_path))
    dataset_loader = image_processing.DatasetLoader(image_paths, dataset_shape, verbose=1)
    (images, labels) = dataset_loader.load()
    images = images.astype("float") / 255.0

    print("[INFO] spliting images for train set and test set ....")
    (trainX, testX, trainY, testY) = train_test_split(images, labels, test_size=test_size, random_state=42)

    print("[INFO] converting labels to vectors ....")
    trainY = LabelBinarizer().fit_transform(trainY)
    testY = LabelBinarizer().fit_transform(testY)

    print("[INFO] loading model ....")
    optimizer = SGD(learning_rate=0.005)
    model = KNeuralNet.build(width=img_width, height=img_height, depth=img_depth, classes=classes)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    print("[INFO] training network ....")
    train = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=batch_size, epochs=epochs, verbose=1)

    print("[INFO] saving model ....")
    model.save("model/k_neural_net.hdf5")

    print("[INFO] evaluating network ....")
    predictions = model.predict(testX, batch_size=batch_size)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=target_names))

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), train.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), train.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), train.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), train.history["val_accuracy"], label="val_acc")
    plt.title("Training loss and accuracy")
    plt.xlabel("Epoch ")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


test_size = 0.1
img_width = 128
img_height = 128
img_depth = 1
classes = 4
batch_size = 16
epochs = 16
target_names = ["1_kor", "2_khor", "3_kur", "4_khur"]
dataset_shape = (4000, 128, 128, 1)
dataset_path = "../khmer_letter_dataset/khmer_letters_air/letters/"

if __name__ == "__main__":
    train(dataset_shape, dataset_path, test_size, img_width, img_height, img_depth, classes, batch_size, epochs, target_names)