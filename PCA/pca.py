# Importer les bibliothèques nécessaires
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Importer la base de données IRIS
iris = datasets.load_iris()

# Afficher le contenu de la base IRIS
print("Features: ", iris.feature_names)
print("Labels: ", iris.target_names)

# Diviser la base de données IRIS en des données d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

# Instancier l'algorithme PCA
pca = PCA(n_components=2)

# Entrainer le PCA sur les données d'IRIS
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Afficher une figure sous la forme d'un nuage de points regroupé par les étiquettes de classe
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, label=iris.target_names)   
plt.xlabel('Première composante principale')
plt.ylabel('Deuxième composante principale')
plt.legend()
plt.show()
