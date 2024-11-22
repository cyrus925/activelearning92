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
n_clusters = 7
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

# Fonction pour copier les fichiers en divisant en train, val et test pour chaque cluster
def copy_files_by_cluster(clusters_dict):
    for cluster_label, image_list in clusters_dict.items():
        print("clusters_dict", clusters_dict.items())
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

        random.shuffle(image_list)
        
        n_total = len(image_list)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        n_test = n_total - n_train - n_val

        train_images = image_list[:n_train]
        val_images = image_list[n_train:n_train + n_val]
        test_images = image_list[n_train + n_val:]

        def move_files(image_list, dest_images_dir, dest_labels_dir):
            for image in image_list:
                shutil.copy(image, dest_images_dir)

                # Copier le fichier d'annotation associé
                label_file = image.replace('.jpg', '.txt').replace('.png', '.txt')
                label_file = label_file.replace('images', 'labels')
                label_path = os.path.join(yolo5label_dir, label_file)
                print("label_file", label_file)
                print("label_path", label_path)
                #shutil.copy(label_path, dest_labels_dir)
                shutil.copy(label_file, dest_labels_dir)

                

        move_files(train_images, train_img_dir, train_label_dir)
        move_files(val_images, val_img_dir, val_label_dir)
        move_files(test_images, test_img_dir, test_label_dir)

        print(f"Cluster {cluster_label}: train ({len(train_images)}), val ({len(val_images)}), test ({len(test_images)}) répartis.")

copy_files_by_cluster(clusters)

print("Séparation des images par clusters et splits train, val, test terminée.")

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
def print_yolov5_commands_for_cluster(cluster_label):
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

print("exécutez les commandes dans le terminal où se situe le yolov5.")
