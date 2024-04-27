import os
import random
import time

import cv2
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
class DBDDataLoader:
    def __init__(self, data_path, image_size, classes):
        self.data_path = data_path
        self.image_size = image_size
        self.classes = classes

    def read_image(self, image_path):
        image = cv2.imread(image_path)
        try:
            resized_image = cv2.resize(image, self.image_size)
        except:
            resized_image = image
        return resized_image

    @staticmethod
    def one_hot_encode_it(vector):
        vector = vector.reshape(-1, 1)
        encoder = OneHotEncoder(sparse=False)
        one_hot_encoded_vector = encoder.fit_transform(vector)
        return one_hot_encoded_vector

    @property
    def load_data(self):
        files = os.listdir(self.data_path)
        ctr = 0
        X_train, Y_train = [], []
        for f in files:
            class_path = os.path.join(self.data_path, f)
            files_per_class = os.listdir(class_path)
            for img_f in files_per_class:
                image_path = os.path.join(class_path, img_f)
                image = self.read_image(image_path)
                if image is not None:
                    X_train.append(image)
                    cls = self.classes.index(f)
                    Y_train.append(cls)
                    ctr += 1
                    if ctr % 500 == 0:
                        print("Image read:", ctr)
                else:
                    print(f"Skipping {image_path} due to read error")
        X_train = np.array(X_train)/255
        Y_train = np.array(Y_train)/255
        Y_train = self.one_hot_encode_it(Y_train)

        # Randomize both X_train and Y_Train
        combined_data = list(zip(X_train, Y_train))
        random.shuffle(combined_data)
        X_train[:], Y_train[:] = zip(*combined_data)

        return X_train, Y_train

data_path = "C:/Users/maram/Downloads/data-5classes/"
image_size = (100, 100)
classes = ['other_activities', 'safe_driving', 'talking_phone', 'texting_phone', 'turning']
dl = DBDDataLoader(data_path, image_size, classes)
X_train, Y_train = dl.load_data
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
print("le nombre total d'images à entrainer:", len(Y_train))
print("le nombre total d'images à tester:", len(Y_test))



# CNN model
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(8, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(16, (3, 3), activation='relu'),
        Flatten(),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

start = time.time()


input_shape = (image_size[0], image_size[1], 3)
num_classes = len(classes)
model = create_model(input_shape, num_classes)
model.summary()
# Train the model
history = model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, Y_test)
print("Test accuracy:", test_acc)



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Faire des prédictions sur les données d'entraînement
Y_train_pred = model.predict(X_train)
y_train_pred = np.argmax(Y_train_pred, axis=1)

# Faire des prédictions sur les données de test
Y_test_pred = model.predict(X_test)
y_test_pred = np.argmax(Y_test_pred, axis=1)

# Calculer les métriques de performance pour les données d'entraînement
train_accuracy = accuracy_score(np.argmax(Y_train, axis=1), y_train_pred)
train_report = classification_report(np.argmax(Y_train, axis=1), y_train_pred, target_names=classes)

# Calculer les métriques de performance pour les données de test
test_accuracy = accuracy_score(np.argmax(Y_test, axis=1), y_test_pred)
test_report = classification_report(np.argmax(Y_test, axis=1), y_test_pred, target_names=classes)

print("Performance sur les données d'entraînement :\n")
print("Accuracy:", train_accuracy)
print("Classification Report:\n", train_report)

print("\nPerformance sur les données de test :\n")
print("Accuracy:", test_accuracy)
print("Classification Report:\n", test_report)




import matplotlib.pyplot as plt
import seaborn as sns
# Calculer la matrice de confusion pour les données d'entraînement
conf_matrix_test = confusion_matrix(np.argmax(Y_test, axis=1), y_test_pred)

# Visualiser la matrice de confusion des données d'entraînement sous forme de heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix - Testing Data')
plt.show()


end = time.time()
print("Total time:", end - start)











