# Afficher quelques exemples d'images pour chaque classe
plt.figure(figsize=(10, 8))
for i, cls in enumerate(classes):
    class_indices = np.where(Y_true_classes == classes.index(cls))[0]
    random_indices = np.random.choice(class_indices, size=3, replace=False)
    for j, idx in enumerate(random_indices):
        plt.subplot(5, 3, i*3 + j + 1)
        plt.imshow(X_test[idx])
        plt.title(cls)
        plt.axis('off')
plt.tight_layout()
plt.show()



# Afficher des exemples d'images pour lesquelles le modèle s'est trompé
incorrect_indices = np.where(Y_pred_classes != Y_true_classes)[0]
plt.figure(figsize=(10, 8))
for i, idx in enumerate(incorrect_indices[:6]):
    plt.subplot(2, 3, i+1)
    plt.imshow(X_test[idx])
    plt.title(f"True: {classes[Y_true_classes[idx]]}\nPred: {classes[Y_pred_classes[idx]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()





import os
import matplotlib.pyplot as plt

# Chemin vers le répertoire contenant les données
data_path = "C:/Users/maram/Downloads/data-5classes/"

# Compter le nombre d'images dans chaque classe
class_image_counts = {}
for cls in classes:
    class_path = os.path.join(data_path, cls)
    if os.path.isdir(class_path):
        class_images = os.listdir(class_path)
        class_image_counts[cls] = len(class_images)
    else:
        print(f"Le répertoire {class_path} n'existe pas.")

# Créer un diagramme à barres pour visualiser le nombre d'images dans chaque classe
plt.figure(figsize=(10, 6))
bars = plt.bar(class_image_counts.keys(), class_image_counts.values(), color='skyblue')
plt.title('Nombre d\'images dans chaque classe')
plt.xlabel('Classes')
plt.ylabel('Nombre d\'images')
plt.xticks(rotation=45)
plt.tight_layout()

# Ajouter les valeurs au-dessus de chaque barre
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='bottom')
# Afficher le diagramme à barres
plt.show()


