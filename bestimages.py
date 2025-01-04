import pickle
import subprocess

# Chemin du fichier pickle dans le répertoire 'output_dataset'
pickle_file_path = 'output_dataset/clusters.pkl'
data_file_path = 'output_dataset/data_split.pkl'

# Ouvrir et charger le fichier pickle
with open(pickle_file_path, 'rb') as f:
    clusters_dict = pickle.load(f)

with open(data_file_path, 'rb') as f:
    data = pickle.load(f)


import random

def select_random_images_from_remaining(data, num_images=100):
    """
    Sélectionne aléatoirement un nombre spécifié d'images à partir de 'remaining_images' dans le dictionnaire 'data'.
    
    :param data: Dictionnaire contenant une clé 'remaining_images' qui liste les chemins des images restantes.
    :param num_images: Le nombre d'images à sélectionner aléatoirement.
    :return: Liste des chemins des images sélectionnées.
    """
    # Extraire la liste des images restantes
    remaining_images = data.get('remaining_images', [])

    # Vérifier que nous avons assez d'images
    if len(remaining_images) < num_images:
        print(f"Attention : Il n'y a que {len(remaining_images)} images disponibles, mais vous en demandez {num_images}.")
        num_images = len(remaining_images)  # Ajuster si moins de 200 images disponibles

    # Sélectionner aléatoirement num_images images
    selected_images = random.sample(remaining_images, num_images)

    # Retourner les images sélectionnées
    return selected_images



# Sélectionner 200 images aléatoires
selected_images = select_random_images_from_remaining(data, num_images=80)

import os
import shutil
output_dir = 'dataset_pipeline'

# Dossier de sortie où les images sélectionnées seront stockées
output_images_dir = os.path.join(output_dir, 'test')

# Créer le dossier 'selected_images' s'il n'existe pas
os.makedirs(output_images_dir, exist_ok=True)


import os
import shutil

def copy_images_to_output(selected_images, output_images_dir):
    """
    Copie les images sélectionnées vers le répertoire de sortie.
    Avant de copier, vérifie que le répertoire de sortie est vide et supprime tous les fichiers s'il y en a.

    :param selected_images: Liste des chemins des images sélectionnées.
    :param output_images_dir: Dossier où les images seront copiées.
    """
    

    # Supprimer tous les fichiers dans le répertoire de sortie s'il n'est pas vide
    for file_name in os.listdir(output_images_dir):
        file_path = os.path.join(output_images_dir, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Copier les images sélectionnées dans le répertoire de sortie
    for image_path in selected_images:
        # Vérifier si le fichier image existe avant de le copier
        if os.path.isfile(image_path):
            # Extraire le nom du fichier de l'image
            image_filename = os.path.basename(image_path)
            
            # Définir le chemin de destination dans le dossier de sortie
            destination_path = os.path.join(output_images_dir, image_filename)
            
            # Copier l'image vers le dossier de sortie
            shutil.copy(image_path, destination_path)
        else:
            print(f"Image non trouvée : {image_path}")

# Copier les images sélectionnées dans le dossier de sortie
copy_images_to_output(selected_images, output_images_dir)




import datetime

# Ajouter un horodatage pour rendre le nom unique
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
project_name="model_yolo5"
experiment_name="experiment"
unique_name = f"detection_results_{current_time}"


# Fonction pour lancer la détection YOLOv5
def run_yolov5_detection(source_dir,unique_name, project_name="model_yolo5", experiment_name="experiment",  weights_filename="best.pt"):
    
    # Définir le chemin vers les poids du modèle
    weights_filepath = os.path.join("yolov5", project_name, experiment_name, "weights", weights_filename)

    # Vérifier si le fichier de poids existe
    if not os.path.exists(weights_filepath):
        print(f"Erreur : Le fichier de poids {weights_filepath} n'existe pas.")
        return

    # Définir la commande à exécuter
    detect_command = (
    f"py -3.12 yolov5/detect.py "
    f"--source {source_dir} "
    f"--weights {weights_filepath} "
    f"--img-size 640 "
    f"--project {project_name} "
    f"--name {unique_name} "
    f"--save-txt "
    f"--save-conf "
)



    # Nom du fichier log
    log_file = f"{experiment_name}_detection_log.txt"

    # Lancer la commande et capturer les logs
    with open(log_file, "w") as log:
        process = subprocess.Popen(detect_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        print(f"Début de la détection pour {experiment_name}. Logs en temps réel :")
        for line in process.stdout:
            print(line, end="")  # Afficher les logs en temps réel dans la console
            log.write(line)      # Enregistrer les logs dans un fichier

        process.wait()  # Attendre la fin de la commande
        if process.returncode == 0:
            print(f"\nDétection terminée avec succès pour {experiment_name}.\n")
        else:
            print(f"\nErreur lors de la détection pour {experiment_name}. Vérifiez les logs.\n")


run_yolov5_detection(output_images_dir,unique_name, project_name,experiment_name)





import re
import os

import os
import pickle

# Chemin des fichiers et dossiers
# Construire le chemin dynamique vers le dossier des labels
img_folder = os.path.join(project_name, unique_name)
txt_folder = os.path.join(img_folder, "labels")
output_dir = 'output_dataset'                        # Dossier contenant les fichiers pickle
data_path = os.path.join(output_dir, "data_split.pkl")
clusters_path = os.path.join(output_dir, "clusters.pkl")

# Chargement des fichiers pickle
with open(data_path, 'rb') as f:
    data_dict = pickle.load(f)
with open(clusters_path, 'rb') as f:
    clusters = pickle.load(f)

# Étape 1 : Lecture des scores de confiance depuis les fichiers YOLOv5
confidence_scores = {}

# Parcourir tous les fichiers .txt dans le dossier
for txt_file in os.listdir(txt_folder):
    if txt_file.endswith('.txt'):
        txt_path = os.path.join(txt_folder, txt_file)
        image_name = os.path.splitext(txt_file)[0] + '.jpg'  # Nom de l'image associée
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        # Extraire les scores de confiance
        scores = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 5:  # Format attendu : class x_center y_center width height confidence
                confidence = float(parts[5])
                scores.append(confidence)
        confidence_scores[image_name] = scores
        


# Deuxième boucle pour attribuer un score de 0 aux images sans fichier .txt
all_images = [img for img in os.listdir(img_folder) if img.endswith('.jpg')]
processed_images = set(confidence_scores.keys())  # Images déjà traitées
unprocessed_images = set(all_images) - processed_images  # Images sans fichier .txt

for image_name in unprocessed_images:
    confidence_scores[image_name] = 0  # Pas de scores associés, liste vide

# Affichage des scores de confiance (facultatif)
#for image, scores in confidence_scores.items():
    #print(f"Image: {image}, Scores de confiance: {scores}")

# Étape 2 : Fonction pour convertir les valeurs en float
def convert_to_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0  # Retourne 0.0 si la conversion échoue

# Étape 3 : Fonction pour filtrer et trier les pires images
def select_worst_images(results, limit=100):
    """
    Sélectionne les pires images en priorisant celles sans prédictions.
    """
    worst_images = []
    images_no_predictions = []

    for image, scores in results.items():
        if not scores:  # Aucune prédiction
            images_no_predictions.append((image, 0))  # Score nul
        else:
            min_score = min(scores)  # Prendre le score minimum
            worst_images.append((image, min_score))

    # Trier les deux listes
    images_no_predictions.sort(key=lambda x: x[1])  # Pas nécessaire ici mais garde un ordre cohérent
    worst_images.sort(key=lambda x: x[1])  # Trier par score croissant

    # Prioriser les images sans prédictions
    combined = images_no_predictions + worst_images
    return combined[:limit]


# Normalisation des noms d'images
def normalize_image_name(image_name):
    return os.path.basename(image_name)





# Fonction pour inverser le dictionnaire des clusters
def invert_clusters(clusters):
    inverted_clusters = {}  # On veut associer chaque image à son cluster
    for cluster_id, image_names in clusters.items():
        for image_name in image_names:
            inverted_clusters[image_name] = cluster_id  # Chaque image est associée à son cluster
    return inverted_clusters

clusters_invert = invert_clusters(clusters)
# Normaliser les noms
worst_images = [(normalize_image_name(img), score) for img, score in select_worst_images(confidence_scores, limit=100)]
clusters_invert = {normalize_image_name(img): cluster for img, cluster in clusters_invert.items()}






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

print("cluster_counts",cluster_counts)
remaining_images = data_dict["remaining_images"]


# Trier les clusters par fréquence (du plus fréquent au moins fréquent)
sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)

# Sélectionner les clusters les plus problématiques (ici on prend les clusters 1 et 2 comme exemple)
problematic_clusters = [cluster_id for cluster_id, count in sorted_clusters[:2]]  # Prendre les 2 clusters les plus fréquents
print("problematic_clusters",problematic_clusters)
# Filtrer les images qui appartiennent aux clusters problématiques dans 'remaining_images'
selected_images = []

for cluster_id in problematic_clusters:

    # Chercher toutes les images dans 'clusters_invert' associées à ce cluster
    for image_name, assigned_cluster in clusters_invert.items():
        if int(assigned_cluster) == cluster_id and f"images\\{image_name}" in remaining_images: #A CHANGER POUR CHAQUE TRUC
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
def clean_image_name(image_name):

    return re.sub(r'^images\\', '', image_name) #A CHANGER POUR CHAQUE TRUC

# Copier chaque image de `selected_images` vers `échantillon/images`
for image_name in selected_images:

    cleaned_image_name = clean_image_name(image_name)
    source_path = os.path.join(source_dir, cleaned_image_name)  # Chemin complet de l'image source
    target_path = os.path.join(target_dir, cleaned_image_name)  # Chemin complet de l'image cible
    
    # Vérifier si l'image existe avant de la copier
    if os.path.exists(source_path):
        shutil.move(source_path, target_path)  
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
