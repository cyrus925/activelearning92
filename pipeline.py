import os
import shutil
import json
import cv2



# Chemins des répertoires d'entrée et de sortie
train_dir = "output_dataset/echantillon"
val_dir = "output_dataset/test"
output_dir = 'dataset_pipeline'

# Définir les répertoires de sortie pour 'train' et 'val'
output_train_dir = os.path.join(output_dir, 'train')
output_val_dir = os.path.join(output_dir, 'val')

# Créer les répertoires 'train' et 'val' dans 'dataset_pipeline' si non existants
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_val_dir, exist_ok=True)

# Liste des classes
classes = ['label', 'pal']

# Fonction pour convertir les coordonnées de la boîte englobante au format YOLO
def convert_bbox(img_size, bbox):
    dw = 1.0 / img_size[1]
    dh = 1.0 / img_size[0]
    x_center = (bbox[0] + bbox[2]) / 2.0
    y_center = (bbox[1] + bbox[3]) / 2.0
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return (x_center * dw, y_center * dh, width * dw, height * dh)

# Fonction pour calculer la boîte englobante à partir des points du polygone
def get_bounding_box_from_polygon(points):
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    xmin = min(x_coords)
    ymin = min(y_coords)
    xmax = max(x_coords)
    ymax = max(y_coords)
    return [xmin, ymin, xmax, ymax]

# Créer le dossier 'preprocessing' s'il n'existe pas
preprocessing_dir = os.path.join(output_dir, 'preprocessing')
os.makedirs(preprocessing_dir, exist_ok=True)

# Créer les sous-dossiers 'train' et 'val' dans 'preprocessing'
train_preprocessing_dir = os.path.join(preprocessing_dir, 'train')
val_preprocessing_dir = os.path.join(preprocessing_dir, 'val')
os.makedirs(train_preprocessing_dir, exist_ok=True)
os.makedirs(val_preprocessing_dir, exist_ok=True)

# Fonction pour déplacer les fichiers
def move_files(src_dir, dst_dir):
    # Utiliser os.walk pour parcourir récursivement tous les sous-dossiers
    for root, dirs, files in os.walk(src_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            
            # Si c'est un fichier image
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy(file_path, os.path.join(dst_dir, filename))
            # Si c'est un fichier JSON
            elif filename.endswith('.json'):
                shutil.copy(file_path, os.path.join(dst_dir, filename))

# Déplacer les fichiers pour le dossier 'val' et 'train'
move_files(val_dir, val_preprocessing_dir)
move_files(train_dir, train_preprocessing_dir)

print("Les fichiers ont été déplacés avec succès !")

# Créer les sous-dossiers 'images' et 'labels' dans 'train' et 'val' pour YOLO
output_image_dir_train = os.path.join(output_train_dir, 'images')
yolo5label_dir_train = os.path.join(output_train_dir, 'labels')
output_image_dir_val = os.path.join(output_val_dir, 'images')
yolo5label_dir_val = os.path.join(output_val_dir, 'labels')

# Créer les répertoires nécessaires
os.makedirs(output_image_dir_train, exist_ok=True)
os.makedirs(yolo5label_dir_train, exist_ok=True)
os.makedirs(output_image_dir_val, exist_ok=True)
os.makedirs(yolo5label_dir_val, exist_ok=True)


import os
import json
import shutil
import cv2





def process_files(train_preprocessing_dir, output_image_dir_train, yolo5label_dir_train, classes):
    # Obtenez la liste des fichiers JSON pour un accès rapide
    json_files = {os.path.splitext(f)[0] for f in os.listdir(train_preprocessing_dir) if f.endswith('.json')}
    
    # Traitez tous les fichiers dans le répertoire
    for file in os.listdir(train_preprocessing_dir):
        if file.endswith(('.jpg', '.png', '.jpeg')):  # Vérifiez les extensions d'image
            image_name = os.path.splitext(file)[0]
            image_path = os.path.join(train_preprocessing_dir, file)
            
            if not os.path.isfile(image_path):
                print(f"Image non trouvée : {image_path}")
                continue
            
            # Si l'image a un fichier JSON associé, traitez-le normalement
            if image_name in json_files:
                json_path = os.path.join(train_preprocessing_dir, image_name + '.json')
                with open(json_path) as f:
                    data = json.load(f)
                    
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Impossible de lire l'image : {image_path}, fichier JSON ignoré.")
                    continue
                
                img_height, img_width = img.shape[:2]
                
                # Préparer les données d'annotations YOLO
                yolo_data = []
                for shape in data['shapes']:
                    label = shape['label']
                    if label not in classes:
                        continue
                    class_id = classes.index(label)
                    
                    # Obtenir la boîte englobante des points du polygone
                    points = shape['points']
                    bbox = get_bounding_box_from_polygon(points)
                    
                    # Convertir la boîte en format YOLO
                    yolo_bbox = convert_bbox((img_height, img_width), bbox)
                    yolo_data.append(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")
                
                # Sauvegarder le fichier d'annotations YOLO si des données sont valides
                if yolo_data:
                    yolo_filename = image_name + '.txt'
                    yolo_filepath = os.path.join(yolo5label_dir_train, yolo_filename)
                    with open(yolo_filepath, 'w') as yolo_file:
                        yolo_file.writelines(yolo_data)
                
                # Copier l'image dans le dossier des images de sortie
                output_image_path = os.path.join(output_image_dir_train, file)
                if not os.path.exists(output_image_path):
                    shutil.copy(image_path, output_image_dir_train)
            
            # Si l'image n'a pas de JSON, copiez-la directement
            else:
                print(f"Image sans JSON trouvée : {image_path}")
                output_image_path = os.path.join(output_image_dir_train, file)
                if not os.path.exists(output_image_path):
                    shutil.copy(image_path, output_image_dir_train)

    print("Traitement terminé. Toutes les images ont été copiées dans le dossier train.")





process_files(train_preprocessing_dir, output_image_dir_train, yolo5label_dir_train, classes)
process_files(val_preprocessing_dir, output_image_dir_val, yolo5label_dir_val, classes)

print("Les fichiers d'entraînement ont été traités et copiés avec succès !")


def verifier_doublons(train_dir, val_dir):
    train_images = set(os.listdir(train_dir))
    val_images = set(os.listdir(val_dir))
    
    doublons = train_images.intersection(val_images)
    if doublons:
        print(f"Attention : Les fichiers suivants sont présents dans les deux ensembles : {doublons}")
    else:
        print("Aucun doublon détecté entre les ensembles d'entraînement et de validation.")

# Vérifiez les doublons
verifier_doublons(output_image_dir_train, output_image_dir_val)


import yaml

# Nombre de classes
nc = 2  
names = ['label', 'pal']


# Créer le contenu YAML
yaml_content = {
    'train': os.path.join('..', output_train_dir).replace(os.sep, '\\'),
    'val': os.path.join('..', output_val_dir).replace(os.sep, '\\'),
    'nc': nc,
    'names': names
}


# Définir le chemin du fichier YAML
yaml_path = os.path.join(output_dir, 'dataset.yaml')

# Sauvegarder le fichier YAML
with open(yaml_path, 'w') as yaml_file:
    yaml.dump(yaml_content, yaml_file, default_flow_style=False)



import os
import subprocess

# Chemin vers le fichier YAML
current_directory = os.getcwd()
yaml_filepath = os.path.join(current_directory, output_dir, "dataset.yaml")

from configparser import ConfigParser
config = ConfigParser()
config.read("config.ini")
epochs=int(config["training"]["epochs"])

def run_yolov5_training(yaml_filepath,epochs =5, project_name="model_yolo5", experiment_name="experiment"):
    install_command = f"py -3.12 -m pip install -r yolov5/requirements.txt"
    subprocess.run(install_command, shell=True, check=True)
    # Définir la commande à exécuter
    train_command = (
        f"py -3.12 train.py --img 640 --batch-size 16 --epochs {epochs} "
        f"--data {yaml_filepath} --weights yolov5s.pt "
        f"--project {project_name} "
        f"--name {experiment_name} --exist-ok --device cpu --save-period 5"
    )

    # Changer le répertoire courant vers le dossier YOLOv5
    os.chdir("yolov5")


    # Nom du fichier log
    log_file = f"{experiment_name}_training_log.txt"

    # Lancer la commande et capturer les logs
    with open(log_file, "w") as log:
        process = subprocess.Popen(train_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        print(f"Début de l'entraînement pour {experiment_name}. Logs en temps réel :")
        for line in process.stdout:
            print(line, end="")  # Afficher les logs en temps réel dans la console
            log.write(line)      # Enregistrer les logs dans un fichier
        
        process.wait()  # Attendre la fin de la commande
        if process.returncode == 0:
            print(f"\nEntraînement terminé avec succès pour {experiment_name}.\n")
        else:
            print(f"\nErreur lors de l'entraînement pour {experiment_name}. Vérifiez les logs.\n")

    # Revenir au répertoire de travail initial
    os.chdir("..")




import os

def delete_cache_files(directory):
    # Parcours du dossier et de ses sous-dossiers
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Vérifie si le fichier est un fichier cache
            if file.endswith('labels.cache'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)  # Supprime le fichier
                    print(f"Supprimé : {file_path}")
                except Exception as e:
                    print(f"Erreur lors de la suppression de {file_path}: {e}")

delete_cache_files(output_dir)






# Appeler la fonction d'entraînement
run_yolov5_training(yaml_filepath,epochs=epochs)



import pandas as pd 
import re


# Fonction pour extraire les résultats de validation du fichier log dans un DataFrame
def extract_validation_results_from_log(experiment_name="experiment"):
    log_file_path = f"yolov5/{experiment_name}_training_log.txt"

    """Extraire les résultats du fichier log de validation dans un DataFrame"""
    with open(log_file_path, "r") as log_file:
        content = log_file.read()

    # Regexp pour extraire les résultats de la table (Class, Images, Instances, P, R, mAP50, mAP50-95)
    pattern = re.compile(r"(\S+)\s+(\d+)\s+(\d+)\s+([0-1]?\d*\.?\d+)\s+([0-1]?\d*\.?\d+)\s+([0-1]?\d*\.?\d+)\s+([0-1]?\d*\.?\d+)")

    # Trouver toutes les correspondances dans le contenu du fichier
    matches = pattern.findall(content)

    # Créer un DataFrame à partir des correspondances
    df = pd.DataFrame(matches, columns=["Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95"])

    # Convertir les colonnes numériques en float pour l'analyse
    df["Images"] = df["Images"].astype(int)
    df["Instances"] = df["Instances"].astype(int)
    df["P"] = df["P"].astype(float)
    df["R"] = df["R"].astype(float)
    df["mAP50"] = df["mAP50"].astype(float)
    df["mAP50-95"] = df["mAP50-95"].astype(float)

    return df


# Extraire les résultats de la validation du fichier log
results = extract_validation_results_from_log()


# Filtrer les lignes pour 'label' et 'pal'
filtered_rows = results[results["Class"].isin(["label", "pal"])]

# Filtrer les lignes pour 'all' et sélectionner la dernière
last_all = results[results["Class"] == "all"].iloc[-1]

# Ajouter la dernière ligne de 'all' aux lignes filtrées
result_df = pd.concat([filtered_rows, last_all.to_frame().T], ignore_index=True)

# Sauvegarder le résultat dans un fichier CSV
from datetime import datetime

# Générer un nom de fichier avec un timestamp
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Format : année-mois-jour_heure-minute-seconde
file_name = f'results-{timestamp}.csv'

results.to_csv(file_name, index=False)

print(f"Fichier sauvegardé sous : {file_name}")


# Fonction pour vider le contenu d'un dossier
def vider_dossier(dossier):
    if os.path.exists(dossier):
        # Supprime tous les fichiers et sous-dossiers
        for fichier in os.listdir(dossier):
            chemin = os.path.join(dossier, fichier)
            if os.path.isfile(chemin) or os.path.islink(chemin):
                os.unlink(chemin)  # Supprimer les fichiers ou les liens symboliques
            elif os.path.isdir(chemin):
                shutil.rmtree(chemin)  # Supprimer les sous-dossiers
        print(f"Le dossier '{dossier}' a été vidé.")
    else:
        print(f"Le dossier '{dossier}' n'existe pas.")


dossier1 = os.path.join(train_dir, "images")
dossier2 = os.path.join(train_dir, "labels")



vider_dossier(dossier1)
vider_dossier(dossier2)



