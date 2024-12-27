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

# Traiter les fichiers JSON et images dans val_preprocessing_dir
for file in os.listdir(val_preprocessing_dir):
    if file.endswith('.json'):
        json_path = os.path.join(val_preprocessing_dir, file)
        
        with open(json_path) as f:
            data = json.load(f)
            image_path = os.path.join(val_preprocessing_dir, data['imagePath'])
            
            # Vérifier si l'image existe
            if not os.path.isfile(image_path):
                print(f"Image non trouvée pour : {data['imagePath']}, fichier JSON ignoré.")
                continue
            
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
                yolo_filename = os.path.splitext(file)[0] + '.txt'
                yolo_filepath = os.path.join(yolo5label_dir_val, yolo_filename)
                with open(yolo_filepath, 'w') as yolo_file:
                    yolo_file.writelines(yolo_data)
                
                # Copier l'image dans le dossier des images de sortie
                shutil.copy(image_path, output_image_dir_val)

print("Les fichiers de validation ont été traités et copiés avec succès !")

# Traiter les fichiers JSON et images dans train_preprocessing_dir
for file in os.listdir(train_preprocessing_dir):
    if file.endswith('.json'):
        json_path = os.path.join(train_preprocessing_dir, file)
        
        with open(json_path) as f:
            data = json.load(f)
            image_path = os.path.join(train_preprocessing_dir, data['imagePath'])
            
            # Vérifier si l'image existe
            if not os.path.isfile(image_path):
                print(f"Image non trouvée pour : {data['imagePath']}, fichier JSON ignoré.")
                continue
            
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
                yolo_filename = os.path.splitext(file)[0] + '.txt'
                yolo_filepath = os.path.join(yolo5label_dir_train, yolo_filename)
                with open(yolo_filepath, 'w') as yolo_file:
                    yolo_file.writelines(yolo_data)
                
                # Copier l'image dans le dossier des images de sortie
                shutil.copy(image_path, output_image_dir_train)




print("Les fichiers d'entraînement ont été traités et copiés avec succès !")





shutil.rmtree(preprocessing_dir)







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

# Fonction pour lancer l'entraînement YOLOv5
def run_yolov5_training(yaml_filepath, project_name="model_yolo5", experiment_name="experiment"):
    # Définir la commande à exécuter
    train_command = (
        f"python train.py --img 640 --batch-size 16 --epochs 1 "
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



# Appeler la fonction d'entraînement
run_yolov5_training(yaml_filepath)


import os
import subprocess

# Fonction pour lancer la validation YOLOv5
def run_yolov5_validation(yaml_filepath, project_name="model_yolo5", experiment_name="experiment", weights_filename="best.pt"):
    # Définir le chemin vers les poids du modèle
    weights_filepath = os.path.join("yolov5", project_name, experiment_name, "weights", weights_filename)
    
    # Vérifier si le fichier de poids existe
    if not os.path.exists(weights_filepath):
        print(f"Erreur : Le fichier de poids {weights_filepath} n'existe pas.")
        return
    
    # Définir la commande à exécuter
    val_command = (
        f"python yolov5/val.py --weights {weights_filepath} --img 640 "
        f"--data {yaml_filepath} "
    )


    # Nom du fichier log
    log_file = f"{experiment_name}_validation_log.txt"

    # Lancer la commande et capturer les logs
    with open(log_file, "w") as log:
        process = subprocess.Popen(val_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        print(f"Début de la validation pour {experiment_name}. Logs en temps réel :")
        for line in process.stdout:
            print(line, end="")  # Afficher les logs en temps réel dans la console
            log.write(line)      # Enregistrer les logs dans un fichier

        process.wait()  # Attendre la fin de la commande
        if process.returncode == 0:
            print(f"\nValidation terminée avec succès pour {experiment_name}.\n")
        else:
            print(f"\nErreur lors de la validation pour {experiment_name}. Vérifiez les logs.\n")



# Appeler la fonction de validation
run_yolov5_validation(yaml_filepath)

import pandas as pd 
import re


# Fonction pour extraire les résultats de validation du fichier log dans un DataFrame
def extract_validation_results_from_log(experiment_name="experiment"):
    log_file_path = f"{experiment_name}_validation_log.txt"

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

print(results)

results.to_csv('results.csv', index=False)



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
