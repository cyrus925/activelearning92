import json
import os

def convert_to_yolo_format(points, img_width, img_height):
    """Convertit un polygone en boîte englobante et normalise les coordonnées pour YOLO."""
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    # Calcul de la boîte englobante (xmin, ymin, xmax, ymax)
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    # Calcul du centre (x, y) et des dimensions (largeur, hauteur) de la boîte
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    x_center = x_min + bbox_width / 2
    y_center = y_min + bbox_height / 2
    
    # Normalisation des coordonnées par la taille de l'image
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    bbox_width_norm = bbox_width / img_width
    bbox_height_norm = bbox_height / img_height
    
    return x_center_norm, y_center_norm, bbox_width_norm, bbox_height_norm

def labelme_to_yolo(json_path, output_txt_path, img_width, img_height, class_mapping):
    """Convertit un fichier JSON Labelme en format YOLO et écrit dans un fichier texte."""
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    
    # Ouvrir un fichier texte pour écrire les annotations YOLO
    with open(output_txt_path, 'w') as out_file:
        for shape in annotations['shapes']:
            label = shape['label']
            points = shape['points']
            
            # Convertir les points en boîte englobante YOLO
            x_center, y_center, bbox_width, bbox_height = convert_to_yolo_format(points, img_width, img_height)
            
            # Obtenir l'ID de la classe à partir du mapping
            class_id = class_mapping[label]
            
            # Écrire l'annotation au format YOLO
            out_file.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

# Exemple d'utilisation

json_path = './labels-pals_all/passageCentre_Bloc1_Camera3_1716735587_1716735587_992_10.json'
output_txt_path = './labels-pals_all/passageCentre_Bloc1_Camera3_1716735587_1716735587_992_10.txt'


# Dimensions de l'image (doivent correspondre à celles de l'image annotée)
img_width = 1920  # Largeur de l'image en pixels
img_height = 1080  # Hauteur de l'image en pixels

# Mapping des classes (à adapter en fonction des classes que YOLO doit apprendre)
class_mapping = {
    'label': 0,  # Par exemple, "label" devient la classe 0
    'pal': 1     # "pal" devient la classe 1
}

# Convertir les annotations Labelme en annotations YOLO
labelme_to_yolo(json_path, output_txt_path, img_width, img_height, class_mapping)
