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


import numpy as np

# Calculer les statistiques de base des valeurs de pixels dans les images
mean_pixels = np.mean(X_train)
std_pixels = np.std(X_train)
median_pixels = np.median(X_train)
min_pixels = np.min(X_train)
max_pixels = np.max(X_train)

# Afficher les statistiques
print("Statistiques de base des valeurs de pixels dans les images :")
print("--------------------------------------------------------")
print("Moyenne des pixels :", mean_pixels)
print("Écart-type des pixels :", std_pixels)
print("Médiane des pixels :", median_pixels)
print("Minimum des pixels :", min_pixels)
print("Maximum des pixels :", max_pixels)
