# 🧠 Classification d’images avec un réseau neuronal dense (DNN)

##  Objectif
Construire un modèle DNN simple pour classer des images de vêtements (Fashion MNIST) en 10 catégories. Ce prototype vise à valider un pipeline IA complet avant d’envisager des architectures plus avancées.
##  Données
- Images 28x28 en niveaux de gris
- 10 classes (vêtements)
- Normalisation des pixels (valeurs entre 0 et 1)
- Labels entiers (0 à 9)
##  Modèle
- Architecture : Flatten → Dense(128, ReLU) → Dense(10, Softmax)
- Compilation : Optimiseur Adam, perte `sparse_categorical_crossentropy`, métrique `accuracy`
##  Entraînement
- 15 époques avec validation split (20%)
- Callback EarlyStopping pour éviter le surentraînement
- Sauvegarde du modèle
##  Résultats
- Précision validation : ~86%
- Visualisation des courbes de précision et de perte avec Matplotlib
##  Hyperparamètres
- Neurones : 128
- Activation : ReLU
- Optimiseur : Adam
- Perte : sparse categorical crossentropy
- Époques : 15
- Validation : 20%
- Callback : EarlyStopping
##  Fonction de perte
`sparse_categorical_crossentropy` est adaptée aux labels entiers, contrairement à `categorical_crossentropy` qui nécessite un encodage one-hot.
##  Conclusion
Le modèle répond aux exigences du brief : il est simple, reproductible, performant (>80%) et constitue une base solide pour des modèles plus avancés.
