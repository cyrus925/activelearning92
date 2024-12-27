import os
import json
import shutil
import cv2
import numpy as np
import random
import yaml
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ------------------------- Étape 1 : Traitement des annotations YOLO -------------------------
# Demander à l'utilisateur de fournir le chemin du dossier contenant les données brutes
data_dir = "labels-pals_images"

# Vérifier si le chemin existe
if not os.path.isdir(data_dir):
    print(f"Erreur : Le dossier '{data_dir}' n'existe pas. Veuillez vérifier le chemin.")
    exit(1)

# Définir les répertoires de sortie
output_dir = 'output_dataset'

# Créer les dossiers de sortie s'ils n'existent pas
os.makedirs(output_dir, exist_ok=True)






# ------------------------- Étape 2 : Extraction des features avec ResNet50 -------------------------
# Charger le modèle ResNet50 pré-entraîné sans la dernière couche
base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
model = Model(inputs=base_model.input, outputs=base_model.output)

# Fonction pour extraire les features d'une image
def extract_features(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image)
    return features.flatten()

# Récupérer les chemins des images

image_paths = [os.path.join(data_dir, img) for img in os.listdir(data_dir) if img.endswith((".jpg", ".png"))]

# Extraire les features pour toutes les images
features = np.array([extract_features(img) for img in image_paths])

# Sauvegarder les features dans un fichier .npy
np.save(os.path.join(output_dir, "image_features.npy"), features)

print("Extraction des features terminée.")

# ------------------------- Étape 3 : Réduction des features avec PCA -------------------------

pca = PCA(n_components=0.95)
reduced_features = pca.fit_transform(features)
print("Dimensions après PCA :", reduced_features.shape)

# ------------------------- Étape 4 : Clustering avec KMeans -------------------------
n_clusters = 10

kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(reduced_features)
labels = kmeans.labels_

print("Clustering terminé.")
print(f"Labels de cluster pour chaque image : {labels}")

# ------------------------- Étape 5 : Séparation des images par cluster et split (train, val, test) -------------------------

clusters = {}
for img, cluster in zip(image_paths, labels):
    if cluster not in clusters:
        clusters[cluster] = []
    clusters[cluster].append(img)




# ------------------------- Étape : Sélection des images de test communes -------------------------

def select_common_test_images(clusters_dict, test_proportion=0.15):
    """
    Sélectionne un ensemble commun d'images de test proportionnellement à la taille des clusters.
    """
    # Calculer le nombre total d'images pour les tests
    total_images = sum(len(images) for images in clusters_dict.values())
    n_test_total = int(total_images * test_proportion)

    # Calculer le nombre d'images de test pour chaque cluster
    cluster_test_counts = {
        cluster: int(len(images) / total_images * n_test_total)
        for cluster, images in clusters_dict.items()
    }

    # Ajuster les décalages pour que la somme totale corresponde exactement à `n_test_total`
    total_assigned = sum(cluster_test_counts.values())
    adjustment = n_test_total - total_assigned
    for cluster in sorted(cluster_test_counts.keys(), key=lambda x: -len(clusters_dict[x])):
        if adjustment == 0:
            break
        cluster_test_counts[cluster] += 1
        adjustment -= 1

    # Construire un ensemble commun d'images de test
    common_test_images = []
    updated_clusters = {}

    for cluster, images in clusters_dict.items():
        random.shuffle(images)
        n_test = cluster_test_counts[cluster]
        common_test_images.extend(images[:n_test])
        updated_clusters[cluster] = images[n_test:]

    return common_test_images, updated_clusters

# ------------------------- Application des fonctions -------------------------

# Sélection des images de test communes

# Sélectionner 10 % comme échantillon global
sampled_images, clusters_after_sampling = select_common_test_images(clusters, test_proportion=0.10)

# Sélectionner 20 % des images restantes comme ensemble de test
test_images, remaining_clusters = select_common_test_images(clusters_after_sampling, test_proportion=0.20)


# Fonction pour créer les dossiers 'images' et 'labels' vides
def create_dirs(image_list, target_dir):
    """
    Crée les répertoires 'images' et 'labels' vides dans le dossier cible.
    :param image_list: Liste des chemins des images (pour créer le dossier images).
    :param target_dir: Répertoire cible pour créer les dossiers 'images' et 'labels'.
    """
    # Créer les dossiers images et labels
    os.makedirs(os.path.join(target_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "labels"), exist_ok=True)
    
    # Créer un fichier vide dans 'labels' pour chaque image (si tu veux créer un fichier vide pour chaque image)
    for image_path in image_list:
        image_name = os.path.basename(image_path)
        target_image_path = os.path.join(target_dir, "images", image_name)
        shutil.copy(image_path, target_image_path)


# Dossier pour l'échantillon
sample_dir = os.path.join(output_dir, "echantillon")
create_dirs(sampled_images, sample_dir)

# Dossier pour le test
test_dir = os.path.join(output_dir, "test")
create_dirs(test_images, test_dir)

print("Les dossiers 'images' et 'labels' ont été créés dans les dossiers respectifs.")



# Identifie les images restantes (celles qui ne sont ni dans l'échantillon, ni dans le test)
remaining_images = []
for cluster, images in remaining_clusters.items():
    remaining_images.extend(images)

# Créer le répertoire 'remaining_images' s'il n'existe pas
remaining_images_dir = os.path.join(output_dir, "remaining_images")
os.makedirs(remaining_images_dir, exist_ok=True)

# Copier les images restantes dans le répertoire 'remaining_images'
for image_path in remaining_images:
    image_name = os.path.basename(image_path)
    target_image_path = os.path.join(remaining_images_dir, image_name)
    shutil.copy(image_path, target_image_path)

print(f"{len(remaining_images)} images restantes ont été copiées dans le dossier 'remaining_images'.")


import pickle




data_split = {
    "train": sampled_images,  # Les images échantillonnées pour l'entraînement
    "remaining_images": remaining_images,  # Les images restantes après l'échantillonnage
    "test": test_images  # Les images sélectionnées comme test
}

# Sauvegarder le dictionnaire dans un fichier pickle
clusters_path = os.path.join(output_dir, "clusters.pkl")
data_path = os.path.join(output_dir, "data_split.pkl")
with open(clusters_path, "wb") as f:
    pickle.dump(clusters, f)
with open(data_path, "wb") as f:
    pickle.dump(data_split, f)

print(f"Dictionnaire des clusters sauvegardé dans : {clusters_path}")
print(f"Dictionnaire data_split sauvegardé dans : {data_path}")
