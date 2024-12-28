import os
import shutil
import random

def copy_random_files(src_dir, dest_dir, proportion=0.30):
    """
    Copie aléatoirement un pourcentage (proportion) de fichiers .jpg et .json depuis le répertoire source
    vers le répertoire de destination.
    """
    # Vérifier que le répertoire source existe
    if not os.path.exists(src_dir):
        print(f"Le répertoire source {src_dir} n'existe pas.")
        return
    
    
    # Créer le répertoire de destination s'il n'existe pas
    os.makedirs(dest_dir, exist_ok=True)


    # Compter le nombre de fichiers dans le répertoire source
    num_files_in_src = len([f for f in os.listdir(dest_directory) if os.path.isfile(os.path.join(src_dir, f))])
    print(f"Nombre de fichiers dans le répertoire source : {num_files_in_src}")


    # Réinitialiser le répertoire de destination : supprimer son contenu s'il existe
    if os.path.exists(dest_dir):
        # Supprimer les fichiers dans le répertoire de destination
        for filename in os.listdir(dest_dir):
            file_path = os.path.join(dest_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        print(f"Le répertoire de destination {dest_dir} a été réinitialisé.")
    else:
        os.makedirs(dest_dir, exist_ok=True)



    # Liste des fichiers .jpg et .json dans le répertoire source
    jpg_files = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]
    json_files = [f for f in os.listdir(src_dir) if f.endswith('.json')]

    # Combiner les fichiers .jpg et .json
    all_files = jpg_files 

    # Nombre total de fichiers à copier (10% de l'ensemble des fichiers)
    n_files_to_copy = int(len(all_files) * proportion)

    # Sélectionner aléatoirement les fichiers à copier
    files_to_copy = random.sample(all_files, n_files_to_copy)

    # Copier les fichiers sélectionnés
    for file in files_to_copy:
        # Déterminer le chemin source et le chemin de destination
        src_file = os.path.join(src_dir, file)
        dest_file = os.path.join(dest_dir, file)

        # Copier le fichier
        shutil.copy(src_file, dest_file)
    

    num_files_in_src = len([f for f in os.listdir(dest_directory) if os.path.isfile(os.path.join(src_dir, f))])
    print(f"Nombre de fichiers dans le répertoire : {num_files_in_src}")


# Exemple d'utilisation
src_directory = "labels-pals_all"  # Remplacez par le chemin réel de votre dataset
dest_directory = "labels-pals_images"  # Remplacez par le chemin de destination

copy_random_files(src_directory, dest_directory)
