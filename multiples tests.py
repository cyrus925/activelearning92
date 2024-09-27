import os

def run_yolov5_train(num_tests, batch_size):
    # Liste pour stocker les valeurs d'époques
    epoch_values = []

    # Demande à l'utilisateur de saisir les valeurs d'époques pour chaque test
    for i in range(num_tests):
        epochs = int(input(f"Entrez le nombre d'époques pour le test {i + 1}: "))
        epoch_values.append(epochs)

    # Chemin vers votre script de formation YOLOv5
    yolov5_train_script = "path/to/yolov5/train.py"

    # Autres arguments pour le script de formation
    data = "C:/5A/Projet/datasets/labels-pals_all"  # chemin vers votre fichier de données
    cfg = "path/to/your/model.yaml"  # chemin vers votre configuration de modèle
    weights = "yolov5s.pt"  # poids de départ, vous pouvez spécifier les vôtres


    for i, epochs in enumerate(epoch_values):
        command = f"python {yolov5_train_script} --img 640 --batch {batch_size} --epochs {epochs} --data {data} --cfg {cfg} --weights {weights}"
        print(f"Lancement de l'entraînement {i + 1} avec {epochs} époques et un batch size de {batch_size}...")
        os.system(command)
        print(f"Entraînement {i + 1} avec {epochs} époques terminé.")

    print("Tous les entraînements sont terminés.")

# Demande à l'utilisateur le nombre de tests et la taille du batch
num_tests = int(input("Entrez le nombre de tests à effectuer: "))
batch_size = int(input("Entrez la taille du batch: "))

# Lancer l'entraînement avec les paramètres spécifiés
run_yolov5_train(num_tests, batch_size)
