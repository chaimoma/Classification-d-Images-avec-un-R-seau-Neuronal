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
- Précision test : 87.91%
- Perte test : 0.3471
- Visualisation des courbes de précision et de perte avec Matplotlib
##  Hyperparamètres
- Neurones : 128
- Activation : ReLU
- Optimiseur : Adam
- Perte : sparse categorical crossentropy
- Époques : 15
- Validation : 20%
- Callback : EarlyStopping
## Visualisation des performances
Deux graphiques ont été tracés pour suivre l’évolution du modèle pendant l’entraînement :
    Courbe de précision : montre l’évolution de la précision sur les données d’entraînement et de validation à chaque époque. Cela permet de vérifier si le modèle apprend correctement et s’il généralise bien.
    Courbe de perte : indique la diminution de l’erreur (loss) au fil des époques. Une perte qui diminue régulièrement est signe d’un bon apprentissage.
Ces visualisations aident à détecter un éventuel surentraînement (overfitting) ou sous-apprentissage.
##  Fonction de perte
 Choix de la fonction de perte :
  -sparse_categorical_crossentropy est utilisée lorsque les labels sont des entiers (ex. : 0, 1, 2…). Elle est simple à implémenter et évite de transformer les labels.
  -categorical_crossentropy est utilisée quand les labels sont encodés en one-hot (ex. : [0, 0, 1, 0, 0…]). Elle nécessite une étape de prétraitement supplémentaire.

Dans ce projet, les labels sont déjà sous forme d’entiers, donc sparse_categorical_crossentropy est le choix le plus adapté.
##  Conclusion
Le modèle répond aux exigences du brief : il est simple, reproductible, performant (>80%) et constitue une base solide pour des modèles plus avancés.
