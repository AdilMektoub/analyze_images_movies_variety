Dans cette partie, nous avons utilisé OpenCV, Python et l' algorithme de clustering k-means pour trouver les couleurs les plus dominantes dans une image.
##Clustering K-Means
Alors, qu'est-ce que k-means exactement ?

K-means est un algorithme de clustering .

L'objectif est de partitionner n points de données en k clusters. Chacun des n points de données sera attribué à un cluster avec la moyenne la plus proche. La moyenne de chaque cluster est appelée son "centroïde" ou "centre".

Dans l'ensemble, l'application de k-means donne k clusters séparés des n points de données d'origine. Les points de données à l'intérieur d'un cluster particulier sont considérés comme "plus similaires" les uns aux autres que les points de données appartenant à d'autres clusters.

Dans notre cas, nous allons regrouper les intensités de pixels d'une image RVB. Étant donné une image de taille MxN , nous avons donc MxN pixels, chacun composé de trois composantes : Rouge, Vert et Bleu respectivement.

Nous traiterons ces pixels MxN comme nos points de données et les regrouperons à l'aide de k-means.

Les pixels appartenant à un cluster donné seront plus similaires en couleur que les pixels appartenant à un cluster distinct.

- Pour exécuter notre script, Nous n'avons besoin que de deux arguments : --image, qui est le chemin d'accès à l'emplacement de notre image sur le disque, et --clusters, le nombre de clusters que nous souhaitons générer.

### Notre algorithme  
-  nous chargeons notre image hors du disque, puis la convertissons du BGR vers l'espace colorimétrique RVB
- nous remodelons notre image pour qu'elle soit une liste de pixels, plutôt qu'une matrice MxN de pixels
- pour trouver les couleurs les plus dominantes dans une image, nous utilisons l' implémentation scikit-learn de k-means pour éviter de réimplémenter l'algorithme.
- Un appel à la fit()méthode regroupe notre liste de pixels. C'est tout ce qu'il y a à faire pour regrouper nos pixels RVB en utilisant Python et k-means.
