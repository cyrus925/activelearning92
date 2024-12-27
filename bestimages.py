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

def select_random_images_from_remaining(data, num_images=200):
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
selected_images = select_random_images_from_remaining(data, num_images=200)

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

# Fonction pour lancer la détection YOLOv5
def run_yolov5_detection(source_dir,project_name="model_yolo5", experiment_name="experiment",  weights_filename="best.pt"):
    
    # Définir le chemin vers les poids du modèle
    weights_filepath = os.path.join("yolov5", project_name, experiment_name, "weights", weights_filename)

    # Vérifier si le fichier de poids existe
    if not os.path.exists(weights_filepath):
        print(f"Erreur : Le fichier de poids {weights_filepath} n'existe pas.")
        return

    # Définir la commande à exécuter
    detect_command = (
        f"python yolov5/detect.py "
        f"--source {source_dir} "
        f"--weights {weights_filepath} "
        f"--img-size 640 "
        f"--project {project_name} "
        f"--name detection_results "
        f"--save-txt "
        f"--conf-thres 0.07"
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


run_yolov5_detection(source_dir=output_images_dir)
