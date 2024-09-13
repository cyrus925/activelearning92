import json
import os
from PIL import Image, ImageDraw

def load_annotations(json_path):
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    return annotations

def draw_annotations(image_path, annotations):
    """Dessine les polygones sur l'image selon les annotations."""
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    for shape in annotations['shapes']:
        points = shape['points']  # Liste des points du polygone
        label = shape['label']    # Etiquette associée au polygone
        
        # Transformation des coordonnées en tuple (nécessaire pour PIL)
        polygon = [(point[0], point[1]) for point in points]
        
        # Dessiner le polygone sur l'image avec une bordure rouge
        draw.polygon(polygon, outline="red")
        
        print(f"Label: {label}, Points: {polygon}")
    
    # Afficher l'image annotée
    image.show()


# Exemple d'utilisation
json_path = './labels-pals_all/passageCentre_Bloc1_Camera3_1716735587_1716735587_992_10.json'
image_path = './labels-pals_all/passageCentre_Bloc1_Camera3_1716735587_1716735587_992_10.jpg'
annotations = load_annotations(json_path)
draw_annotations(image_path, annotations)




