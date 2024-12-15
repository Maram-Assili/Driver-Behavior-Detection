# Driver-Behavior-Detection
## Description du Projet
Ce projet a pour objectif de développer un système de détection des comportements des conducteurs en utilisant des réseaux neuronaux convolutionnels (CNN). À travers l'analyse des images provenant de caméras embarquées, le système identifie divers comportements de conduite, tels que la conduite prudente, parler au téléphone au cours de la conduite , envoyer des textos, tourneret autres comportements. L'utilisation de l'architecture VGGNet, reconnue pour son efficacité dans la classification d'images, permet d'atteindre des performances élevées dans la reconnaissance des comportements des conducteurs.

## Objectifs

- Détection Automatique : Identifier et classifier les comportements des conducteurs à partir des images .
- Adaptabilité : Créer un modèle qui peut être ajusté pour différents scénarios de conduite en fonction des données d'entraînement disponibles.
  
## Technologies Utilisées

- Langage de Programmation : Python
- Bibliothèques :
  
1- TensorFlow/Keras : Pour la création et l'entraînement du modèle CNN basé sur VGGNet.
  
2- OpenCV : Pour le traitement des images .

3- NumPy et Pandas : Pour la manipulation et l'analyse des données.

- Ensemble de Données : Un jeu de données contenant des séquences vidéo annotées avec les comportements de conduite observés.
  
## Méthodologie

- Prétraitement des Données : Traitement des images; redimensionnement et normalisation des images pour les adapter au modèle.
- Architecture du Modèle : Utilisation de VGGNet avec des ajustements pour la classification des comportements de conduite. Ce modèle est composé de plusieurs couches convolutionnelles et de couches entièrement connectées.
- Entraînement et Évaluation : Formation du modèle sur les données d'entraînement, suivi d'une évaluation des performances sur les ensembles de validation et de test.
  
## Résultats Attendus

- Un modèle performant capable de détecter et de classifier les comportements des conducteurs avec une haute précision.
- Des visualisations des résultats permettant d'analyser les comportements détectés et d'identifier les tendances à risque.
- Un système pouvant être intégré dans des applications de sécurité routière ou des véhicules autonomes.
