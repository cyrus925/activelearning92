import os
import json
import shutil
import cv2
import numpy as np
import random
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ------------------------- Étape 1 : Traitement des annotations YOLO -------------------------
# Demander à l'utilisateur de fournir le chemin du dossier contenant les données brutes
data_dir = input("Veuillez entrer le chemin du dossier contenant les données brutes fournies par le client (ex: ./labels-pals_all) : ")

# Vérifier si le chemin existe
if not os.path.isdir(data_dir):
    print(f"Erreur : Le dossier '{data_dir}' n'existe pas. Veuillez vérifier le chemin.")
    exit(1)

# Définir les répertoires de sortie
output_dir = 'dataset'
yolo5label_dir = os.path.join(output_dir, 'labels')
output_image_dir = os.path.join(output_dir, 'images')

# Créer les dossiers de sortie s'ils n'existent pas
os.makedirs(yolo5label_dir, exist_ok=True)
os.makedirs(output_image_dir, exist_ok=True)

# Définir les classes
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

# Traiter les fichiers JSON
for file in os.listdir(data_dir):
    if file.endswith('.json'):
        json_path = os.path.join(data_dir, file)
        
        with open(json_path) as f:
            data = json.load(f)
            image_path = os.path.join(data_dir, data['imagePath'])
            
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
                yolo_filepath = os.path.join(yolo5label_dir, yolo_filename)
                with open(yolo_filepath, 'w') as yolo_file:
                    yolo_file.writelines(yolo_data)
                
                # Copier l'image dans le dossier des images de sortie
                shutil.copy(image_path, output_image_dir)

print("Traitement des fichiers JSON terminé. Les images et annotations sont organisées dans le dossier 'dataset'.")

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

image_paths = [os.path.join(output_image_dir, img) for img in os.listdir(output_image_dir) if img.endswith((".jpg", ".png"))]

# Extraire les features pour toutes les images
features = np.array([extract_features(img) for img in image_paths])

# Sauvegarder les features dans un fichier .npy
np.save("dataset/image_features.npy", features)

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
output_base_dir = os.path.join(output_dir, 'cluster')
os.makedirs(output_base_dir, exist_ok=True)

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

# ------------------------- Étape : Copie des fichiers par cluster -------------------------

def copy_files_with_common_test(clusters_dict, common_test_images):
    """
    Copie les fichiers pour chaque cluster en train, val et test.
    Les images de test communes sont exclues de train/val.
    """
    for cluster_label, image_list in clusters_dict.items():
        cluster_dir = os.path.join(output_base_dir, f"cluster_{cluster_label}")
        train_img_dir = os.path.join(cluster_dir, "train", "images")
        val_img_dir = os.path.join(cluster_dir, "val", "images")
        test_img_dir = os.path.join(cluster_dir, "test", "images")
        train_label_dir = os.path.join(cluster_dir, "train", "labels")
        val_label_dir = os.path.join(cluster_dir, "val", "labels")
        test_label_dir = os.path.join(cluster_dir, "test", "labels")

        # Création des sous-dossiers
        for path in [train_img_dir, val_img_dir, test_img_dir, train_label_dir, val_label_dir, test_label_dir]:
            os.makedirs(path, exist_ok=True)

        # Exclure les images de test communes de train/val
        image_list = [img for img in image_list if img not in common_test_images]

        random.shuffle(image_list)
        n_total = len(image_list)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)

        train_images = image_list[:n_train]
        val_images = image_list[n_train:n_train + n_val]

        def move_files(image_list, dest_images_dir, dest_labels_dir):
            for image in image_list:
                shutil.copy(image, dest_images_dir)

                # Copier le fichier d'annotation associé
                label_file = image.replace('.jpg', '.txt').replace('.png', '.txt')
                label_file = label_file.replace('images', 'labels')
                shutil.copy(label_file, dest_labels_dir)

        # Copier train, val
        move_files(train_images, train_img_dir, train_label_dir)
        move_files(val_images, val_img_dir, val_label_dir)

        # Copier les fichiers de test commun dans chaque cluster
        move_files(common_test_images, test_img_dir, test_label_dir)

        print(f"Cluster {cluster_label}: train ({len(train_images)}), val ({len(val_images)}), test ({len(common_test_images)}) répartis.")

def validate_splits(clusters_dict, common_test_images):
    """
    Valide que les ensembles train, val, et test sont disjoints.
    """
    train_images = set()
    val_images = set()
    test_images = set(common_test_images)

    for cluster, images in clusters_dict.items():
        cluster_train = set(images[:int(0.7 * len(images))])
        cluster_val = set(images[int(0.7 * len(images)):int(0.85 * len(images))])
        
        # Intersection check
        assert train_images.isdisjoint(cluster_train)
        assert val_images.isdisjoint(cluster_val)
        assert test_images.isdisjoint(cluster_train.union(cluster_val))
        
        # Add to global sets
        train_images.update(cluster_train)
        val_images.update(cluster_val)

    print("Validation : les ensembles sont bien disjoints.")

# ------------------------- Application des fonctions -------------------------

# Sélection des images de test communes
common_test_images, updated_clusters = select_common_test_images(clusters)

# Copier les fichiers avec le test commun
copy_files_with_common_test(updated_clusters, common_test_images)

validate_splits(updated_clusters, common_test_images)
# ------------------------- Étape 6 : Création des fichiers YAML pour chaque cluster -------------------------
nc = 2  # Nombre de classes
names = ['label', 'pal']

def create_yaml_for_cluster(cluster_label):
    cluster_dir = os.path.join(output_base_dir, f"cluster_{cluster_label}")
    yaml_content = f"""
train: ..\\{os.path.join('dataset', 'cluster', f'cluster_{cluster_label}', 'train', 'images').replace(os.sep, '\\\\')}
val: ..\\{os.path.join('dataset', 'cluster', f'cluster_{cluster_label}', 'val', 'images').replace(os.sep, '\\\\')}
test: ..\\{os.path.join('dataset', 'cluster', f'cluster_{cluster_label}', 'test', 'images').replace(os.sep, '\\\\')}

nc: {nc}

names: {names}
"""
    yaml_path = os.path.join(cluster_dir, "dataset.yaml")
    with open(yaml_path, 'w') as yaml_file:
        yaml_file.write(yaml_content.strip())
    print(f"Fichier dataset.yaml créé pour le cluster {cluster_label}")

for cluster_label in range(n_clusters):
    create_yaml_for_cluster(cluster_label)

# ------------------------- Étape 7 : Génération des commandes pour YOLOv5 -------------------------
""" def print_yolov5_commands_for_cluster(cluster_label):
    yaml_filepath = os.path.join("..", "dataset", "cluster", f"cluster_{cluster_label}", "dataset.yaml")
    command = [
        "python", "train.py",
        "--img", "640",
        "--batch-size", "16",
        "--epochs", "50",
        "--data", yaml_filepath,
        "--weights", "yolov5s.pt",
        "--project", f"model2_yolo5_cluster_{cluster_label}",
        "--name", f"experiment_cluster_{cluster_label}",
        "--exist-ok",
        "--device", "cpu",
        "--save-period", "5"
    ]
    
    print(f"Commande à exécuter pour le cluster {cluster_label} :")
    print(" ".join(command))
    print()

for cluster_label in range(n_clusters):
    print_yolov5_commands_for_cluster(cluster_label)

print("ces commandes vont être executer dans le terminal où se situe le yolov5.") """



import subprocess

def run_yolov5_training(cluster_label):
    # Définir le chemin vers le fichier YAML
    yaml_filepath = os.path.join("..", "dataset", "cluster", f"cluster_{cluster_label}", "dataset.yaml")
    
    # Définir la commande à exécuter
    train_command = (
        f"python train.py --img 640 --batch-size 16 --epochs 100 "
        f"--data {yaml_filepath} --weights yolov5s.pt "
        f"--project model2_yolo5_cluster_{cluster_label} "
        f"--name experiment_cluster_{cluster_label} --exist-ok --device cpu --save-period 5"
    )
    
    # Changer le répertoire courant vers le dossier YOLOv5
    os.chdir("yolov5")
    
    # Nom du fichier log
    log_file = f"training_log_cluster_{cluster_label}.txt"
    
    # Lancer la commande et capturer les logs
    with open(log_file, "w") as log:
        process = subprocess.Popen(train_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        print(f"Début de l'entraînement pour le cluster {cluster_label}. Logs en temps réel :")
        for line in process.stdout:
            print(line, end="")  # Afficher les logs en temps réel dans la console
            log.write(line)      # Enregistrer les logs dans un fichier
        
        process.wait()  # Attendre la fin de la commande
        if process.returncode == 0:
            print(f"\nEntraînement terminé avec succès pour le cluster {cluster_label}.\n")
        else:
            print(f"\nErreur lors de l'entraînement pour le cluster {cluster_label}. Vérifiez les logs.\n")


for cluster_label in range(n_clusters):
    run_yolov5_training(cluster_label)