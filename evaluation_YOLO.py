import torch
import cv2
import numpy as np

def evaluate_model(model_path, img_dir, label_dir, img_size=640, conf_threshold=0.25):
    # Charger le modèle YOLOv5
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    model.conf = conf_threshold  # Confiance minimale pour considérer une prédiction

    # Parcourir les images du dossier
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))
        
        # Charger et prédire
        img = cv2.imread(img_path)
        results = model(img, size=img_size)
        
        # Extraire les résultats
        pred_boxes = results.xywh[0].cpu().numpy()  # Prédictions: [x_center, y_center, width, height, confidence, class]
        true_boxes = np.loadtxt(label_path)  # Charger les annotations YOLO

        if true_boxes.size == 0:
            true_boxes = np.empty((0, 5))  # Cas où il n'y a pas d'annotations

        # Comparer les prédictions aux annotations
        iou_threshold = 0.5  # Seuil d'IoU (Intersection over Union)
        for pred in pred_boxes:
            iou_scores = [iou(pred[:4], tb[:4]) for tb in true_boxes]  # Calcul de l'IoU
            max_iou = max(iou_scores) if iou_scores else 0
            
            if max_iou < iou_threshold:
                print(f"Image problématique détectée: {img_name}")
                # Afficher ou enregistrer l'image
                cv2.imshow('Problematic Image', img)
                cv2.waitKey(0)

def iou(box1, box2):
    # Calcul de l'Intersection over Union (IoU)
    x1_min, y1_min = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    x1_max, y1_max = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2

    x2_min, y2_min = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    x2_max, y2_max = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    iou_value = inter_area / (box1_area + box2_area - inter_area)
    return iou_value
