import easyocr
import re
import os
from PIL import Image

# Créer un lecteur EasyOCR
reader = easyocr.Reader(['en'])  # Utilisez 'fr' pour le français, ou 'en' pour l'anglais

def normalize_value(text):
    # Chercher un nombre à 3 chiffres dans le texte
    match = re.search(r'\b\d{3}\b', text)  # Recherche d'un nombre de 3 chiffres
    if match:
        # Convertir le nombre en décimal (ex: 007 -> 0.07, 092 -> 0.92)
        value = int(match.group()) / 100
        return f"{value:.2f}"  # Retourne le nombre normalisé avec 2 décimales
    return text  # Si aucun nombre à 3 chiffres, retourne le texte inchangé

# Dossier contenant les images
image_folder = 'model_yolo5/detection_results5'  # Remplacez par le chemin du dossier d'images

# Dictionnaire pour stocker les résultats
results_dict = {}

# Parcourir toutes les images dans le dossier
for filename in os.listdir(image_folder):
    # Vérifier si le fichier est une image (extensions d'image courantes)
    if filename.lower().endswith(('.jpg')):
        image_path = os.path.join(image_folder, filename)
        
        # Extraire le texte de l'image
        result = reader.readtext(image_path)
        
        # Dictionnaire pour stocker les classes et leurs valeurs
        image_classes = {'label': 0, 'pal': 0}
        
        # Filtrer et traiter les résultats extraits
        for detection in result:
            text = detection[1]  # Extraction du texte
            
            # Recherche de 'label' ou 'pal' suivis de chiffres
            match = re.search(r'^(label|pal)\s?(\d+)', text)
            
            if match:
                class_name = match.group(1)  # Capturer 'label' ou 'pal'
                value = match.group(2)  # Capturer la valeur (chiffres)
                
                # Normaliser la valeur si elle est à 3 chiffres
                if len(value) == 3:
                    normalized_value = int(value) / 100  # Normalisation
                    image_classes[class_name] = normalized_value  # Stocker la classe et sa valeur
                else:
                    image_classes[class_name] = value  # Stocker la classe et sa valeur originale
        
        # Ajouter les résultats de l'image dans le dictionnaire des résultats
        results_dict[filename] = image_classes
        print(f"Résultats de l'image {filename}: {image_classes}")


import pickle
import os
from collections import defaultdict

# Chemin du fichier pickle dans le répertoire 'output_dataset'

# Ouvrir et charger le fichier pickle
output_dir = 'output_dataset'
data_path = os.path.join(output_dir, "data_split.pkl")
clusters_path = os.path.join(output_dir, "clusters.pkl")

with open(data_path, 'rb') as f:
    data_dict = pickle.load(f)
with open(clusters_path, 'rb') as f:
    clusters = pickle.load(f)



# Fonction pour convertir les valeurs en float, et gérer les cas spéciaux ('None', '0', etc.)
def convert_to_float(value):
    try:
        # Essayer de convertir en float, si c'est une chaîne de caractères
        if isinstance(value, str):
            if value.lower() == 'none' or value == '0':  # Cas spécial pour 'None' ou '0'
                return 0.0
            return float(value)  # Conversion en float
        return float(value)  # Conversion en float si ce n'est pas déjà un nombre
    except ValueError:
        return 0.0  # Retourne 0 en cas d'erreur de conversion

# Fonction pour filtrer et trier les pires images
def select_worst_images(results, limit=100):
    worst_images = []
    other_images = []

    # Parcourir les résultats et séparer les images sans prédiction (label=0 et pal=0 ou None)
    for image, predictions in results.items():
        label = predictions.get('label', None)
        pal = predictions.get('pal', None)
        
        # Convertir les valeurs en float
        label = convert_to_float(label)
        pal = convert_to_float(pal)
        
        # Si les prédictions sont égales à 0, on les considère comme "pire"
        if (label == 0 and pal == 0) or (label < 0.5 or pal < 0.5):
            worst_images.append((image, predictions))
        else:
            other_images.append((image, predictions))

    # Trier les autres images (non nulles ou 0) par les prédictions les plus faibles
    other_images.sort(key=lambda x: (convert_to_float(x[1]['label']), convert_to_float(x[1]['pal'])))  # Tri par les valeurs les plus basses de label et pal

    # Combiner les pires images et les autres avec les prédictions faibles, et limiter à "limit" images
    worst_images.extend(other_images[:limit - len(worst_images)])

    return worst_images[:limit]


worst_images = select_worst_images(results_dict, limit=100)



# Fonction pour inverser le dictionnaire des clusters
def invert_clusters(clusters):
    inverted_clusters = {}  # On veut associer chaque image à son cluster
    for cluster_id, image_names in clusters.items():
        for image_name in image_names:
            inverted_clusters[image_name] = cluster_id  # Chaque image est associée à son cluster
    return inverted_clusters

clusters_invert = invert_clusters(clusters)


# Dictionnaire pour compter les occurrences de chaque cluster
cluster_counts = {}

# Parcourir la liste worst_image et vérifier les clusters_invert
for img_name, _ in worst_images:
    for cluster_path, cluster_id in clusters_invert.items():
        # Convertir la clé en chaîne si nécessaire
        if not isinstance(cluster_path, str):
            cluster_path = str(cluster_path)  # Convertir en chaîne de caractères
            
        cluster_name = os.path.basename(cluster_path)  # Extraire le nom du fichier du chemin complet
        
        # Comparer le nom de l'image dans worst_image avec le nom extrait du chemin dans clusters_invert
        if img_name == cluster_name:
            # Convertir la valeur numpy.int32 en int natif
            cluster_id = int(cluster_id)
            
            # Compter les occurrences de chaque cluster
            if cluster_id in cluster_counts:
                cluster_counts[cluster_id] += 1
            else:
                cluster_counts[cluster_id] = 1

print(cluster_counts)
remaining_images = data_dict["remaining_images"]


# Trier les clusters par fréquence (du plus fréquent au moins fréquent)
sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)

# Sélectionner les clusters les plus problématiques (ici on prend les clusters 1 et 2 comme exemple)
problematic_clusters = [cluster_id for cluster_id, count in sorted_clusters[:2]]  # Prendre les 2 clusters les plus fréquents

# Filtrer les images qui appartiennent aux clusters problématiques dans 'remaining_images'
selected_images = []
for cluster_id in problematic_clusters:
    # Chercher toutes les images dans 'clusters_invert' associées à ce cluster
    for image_name, assigned_cluster in clusters_invert.items():
        if assigned_cluster == cluster_id and image_name in remaining_images:
            selected_images.append(image_name)
        # Limiter à 50 images
        if len(selected_images) >= 50:
            break
    if len(selected_images) >= 50:
        break




# Dossier source où les images sont situées (par exemple dans 'remaining_images')
source_dir = 'output_dataset/remaining_images'

# Dossier cible où les images doivent être copiées
target_dir = 'output_dataset/echantillon/images'
import shutil
import re
print(os.getcwd())
def clean_image_name(image_name):
    return re.sub(r'^labels-pals_images\\', '', image_name)

# Copier chaque image de `selected_images` vers `échantillon/images`
for image_name in selected_images:
    # Nettoyer le nom de l'image pour enlever 'labels-pals_images\'
    cleaned_image_name = clean_image_name(image_name)
    
    source_path = os.path.join(source_dir, cleaned_image_name)  # Chemin complet de l'image source
    target_path = os.path.join(target_dir, cleaned_image_name)  # Chemin complet de l'image cible
    
    # Vérifier si l'image existe avant de la copier
    if os.path.exists(source_path):
        shutil.copy(source_path, target_path)  # Copier l'image
        print(f"Image {cleaned_image_name} copiée avec succès.")
    else:
        print(f"Image {cleaned_image_name} non trouvée dans le répertoire source.")


train_images = data_dict['train']

# Supprimer les images de `remaining_images` et les ajouter à `train`
remaining_images = set(data_dict['remaining_images'])  # Convertir en ensemble pour permettre des suppressions rapides
train_images = set(data_dict['train'])                # Convertir en ensemble pour éviter les doublons


for image in selected_images:
    if image in remaining_images:
        remaining_images.remove(image)  # Retirer de remaining_images
        train_images.add(image)         # Ajouter à train

# Remettre les données sous forme de liste
data_dict['remaining_images'] = list(remaining_images)
data_dict['train'] = list(train_images)

# Sauvegarder le fichier data_dict mis à jour
with open(data_path, 'wb') as f:
    pickle.dump(data_dict, f)
