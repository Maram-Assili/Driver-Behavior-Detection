import cv2
import os
import time
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder

class ddbdata_loader:
    def __init__(self, data_path, image_size, classes):
        self.data_path = data_path
        self.image_size = image_size
        self.classes = classes

    def read_image(self, image_path):
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, self.image_size)
         #try:
         #except:
         #   resized_image = image
        return resized_image



    @staticmethod
    def one_hot_encode_it(vector):
        vector = vector.reshape(-1, 1)
        encoder = OneHotEncoder()
        one_hot_encoded_vector = encoder.fit_transform(vector)
        return one_hot_encoded_vector.toarray()

    def load_data(self):
        files = os.listdir(self.data_path)
        ctr = 0
        X_train, Y_train = [], []
        err = 0
        for f in files:
            class_path = f"{self.data_path}{f}/"
            files_per_class = os.listdir(class_path)
            for img_f in files_per_class:
                image_path = f"{self.data_path}{f}/{img_f}"
                try:
                    image = self.read_image(image_path)
                    X_train.append(image)
                    cls = self.classes.index(f)
                    Y_train.append(cls)
                except:
                    err+=1
                    print("err:", err)
                ctr += 1
                if ctr % 100 == 0:
                    print(f"image read: {ctr}/{len(files_per_class)}")
        X_train = np.array(X_train)/255
        Y_train = np.array(Y_train)/255
        Y_train = self.one_hot_encode_it(Y_train)

        # add randomization to both X_train and Y_Train
        combined_data = list(zip(X_train,Y_train))
        random.shuffle(combined_data)
        X_train[:],Y_train[:]=zip(*combined_data)
        return X_train, Y_train

start = time.time()
data_path = "C:/Users/maram/Downloads/data-5classes/"
print(os.listdir(data_path))
image_size = (100,100)
classes = ['other_activities', 'safe_driving', 'talking_phone', 'texting_phone', 'turning']
dl = ddbdata_loader(data_path, image_size, classes)
X_train, Y_train = dl.load_data()

print(len(X_train))
print(len(Y_train))

from sklearn.model_selection import train_test_split
# Divisez les données en ensembles d'entraînement, de test et de validation
X_train, X_remaining, Y_train, Y_remaining = train_test_split(X_train, Y_train, test_size=0.25, random_state=42)
X_test, X_valid, Y_test, Y_valid = train_test_split(X_remaining, Y_remaining, test_size=0.2,random_state=42)


print(len(X_train))
print(len(X_remaining))
print(len(X_valid))

from tensorflow.keras import layers, models, Input, Model

def VGGNet():
    inp = layers.Input((100, 100, 3))
    x = layers.Conv2D(4, 3, 1, activation='relu')(inp)
    x = layers.Conv2D(4, 3, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Conv2D(8, 3, 1, activation='relu')(x)
    x = layers.Conv2D(8, 3, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Conv2D(16, 3, 1, activation='relu')(x)
    x = layers.Conv2D(16, 3, 1, activation='relu')(x)
    x = layers.Conv2D(16, 3, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Conv2D(32, 3, 1, activation='relu')(x)
    x = layers.Conv2D(32, 3, 1, activation='relu')(x)
    x = layers.Conv2D(32, 3, 1, activation='relu')(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(5, activation='softmax')(x)

    model_VGG = models.Model(inputs=inp, outputs=x)

    return model_VGG

model_VGG = VGGNet()
model_VGG.summary()


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

model_VGG.compile(loss=BinaryCrossentropy(),
              optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Entraînement du modèle
history = model_VGG.fit(X_train, Y_train, epochs=10, validation_data=(X_valid, Y_valid))

# Évaluation du modèle
test_loss, test_acc = model_VGG.evaluate(X_test, Y_test)
print("Test accuracy:", test_acc)
print("Loss sur les données de test:", test_loss)


import matplotlib.pyplot as plt

# Extraire l'historique de l'entraînement
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Tracer les courbes de perte
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Tracer les courbes d'exactitude
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

from sklearn.metrics import classification_report, confusion_matrix

# Prédiction sur l'ensemble de test
Y_pred = model_VGG.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Convertir les étiquettes de test en classes
Y_true_classes = np.argmax(Y_test, axis=1)

# Rapport de classification
print("Rapport de classification :")
print(classification_report(Y_true_classes, Y_pred_classes, target_names=classes))

# Matrice de confusion
print("Matrice de confusion :")
conf_matrix = confusion_matrix(Y_true_classes, Y_pred_classes)
print(conf_matrix)



