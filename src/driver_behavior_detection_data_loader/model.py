
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

