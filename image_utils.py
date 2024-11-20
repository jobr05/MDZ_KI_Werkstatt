import os
import cv2
import numpy as np
import imagehash
from PIL import Image
from pathlib import Path
import shutil
import glob
import matplotlib.pyplot as plt  # Matplotlib für die Bildanzeige


def show_images(FOLDER_PATH):
    # Sammle alle Bildpfade
    image_paths = []
    for subdir, dirs, files in os.walk(FOLDER_PATH):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(subdir, file))

    if not image_paths:
        print("Keine Bilder gefunden.")
        return

    # Berechne Grid-Dimensionen
    n_images = len(image_paths)
    n_cols = min(4, n_images)  # Maximal 4 Bilder pro Zeile
    n_rows = (n_images + n_cols - 1) // n_cols

    # Erstelle Grid
    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))

    for idx, img_path in enumerate(image_paths, 1):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax = fig.add_subplot(n_rows, n_cols, idx)
        ax.imshow(img)
        ax.set_title(os.path.basename(img_path))
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def check_duplicates(FOLDER_PATH):
    """
    Erkennt Duplikate basierend auf Bildinhalten durch verschiedene Hash-Methoden.
    Zeigt jedes Duplikat-Paar nur einmal an.
    """
    from PIL import Image
    import imagehash
    import numpy as np
    from collections import defaultdict

    # Set für bereits gezeigte Duplikate
    shown_duplicates = set()

    # Dictionary für verschiedene Hash-Typen
    hash_dict = {
        'average': defaultdict(list),
        'perceptual': defaultdict(list),
        'difference': defaultdict(list)
    }

    image_files = [f for f in os.listdir(FOLDER_PATH)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    print("🔍 Suche nach Duplikaten...")

    # Berechne verschiedene Hashes für jedes Bild
    for img_file in image_files:
        try:
            img_path = os.path.join(FOLDER_PATH, img_file)
            with Image.open(img_path) as img:
                # Konvertiere zu RGB falls notwendig
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Berechne verschiedene Hash-Typen
                avg_hash = str(imagehash.average_hash(img))
                phash = str(imagehash.phash(img))
                dhash = str(imagehash.dhash(img))

                # Speichere alle Hash-Typen
                hash_dict['average'][avg_hash].append(img_file)
                hash_dict['perceptual'][phash].append(img_file)
                hash_dict['difference'][dhash].append(img_file)

        except Exception as e:
            print(f"⚠️ Fehler bei der Verarbeitung von {img_file}: {str(e)}")

    # Finde und zeige Duplikate
    duplicates_found = False

    for hash_type, hash_values in hash_dict.items():
        for hash_value, files in hash_values.items():
            if len(files) > 1:
                # Sortiere die Dateinamen, um eine konsistente Reihenfolge zu gewährleisten
                files = sorted(files)

                # Erstelle einen eindeutigen Key für das Duplikat-Set
                duplicate_key = tuple(sorted(files))

                # Überspringe, wenn dieses Set bereits gezeigt wurde
                if duplicate_key in shown_duplicates:
                    continue

                duplicates_found = True
                shown_duplicates.add(duplicate_key)

                print(f"\n🔍 Duplikate gefunden ({hash_type} hash):")

                # Zeige die Duplikate in einem Grid an
                n_images = len(files)
                n_cols = min(3, n_images)
                n_rows = (n_images + n_cols - 1) // n_cols

                fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))

                for idx, duplicate in enumerate(files, 1):
                    img_path = os.path.join(FOLDER_PATH, duplicate)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    ax = fig.add_subplot(n_rows, n_cols, idx)
                    ax.imshow(img)
                    ax.set_title(duplicate, fontsize=10)
                    ax.axis('off')

                    print(f"- {duplicate}")

                plt.tight_layout()
                plt.show()

    if not duplicates_found:
        print("✅ Keine Duplikate gefunden!")


def check_image_quality(FOLDER_PATH):
    image_files = os.listdir(FOLDER_PATH)
    found_blurry = False
    
    # Bildschärfe prüfen
    for img_file in image_files:
        img_path = os.path.join(FOLDER_PATH, img_file)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 100:  # Schwellenwert für Unschärfe
            found_blurry = True
            print(f"⚠️ Möglicherweise unscharfes Bild gefunden:")
            print(f"- {img_file} (Schärfewert: {laplacian_var:.2f})")

            # Zeige das unscharfe Bild an
            plt.figure(figsize=(3, 3))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            #plt.title(f"Unscharfes Bild: {img_file}\nSchärfewert: {laplacian_var:.2f}")
            plt.axis('off')
            plt.show()
    
    if not found_blurry:
        print("✅ Keine unscharfen Bilder gefunden!")

def analyze_images(FOLDER_PATH):
    print("🔄 Starte Überprüfung auf Duplikate.")
    check_duplicates(FOLDER_PATH)
    print("Überprüfung beendet.")

    print("Überprüfe Bildqualität...")
    check_image_quality(FOLDER_PATH)

