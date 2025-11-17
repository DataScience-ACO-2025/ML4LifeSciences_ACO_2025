import kagglehub
import os
from PIL import Image
import os

# Download latest version
path = kagglehub.dataset_download("joebeachcapital/defungi")

print("Path to dataset files:", path)

for root, dirs, files in os.walk(path):
    print(root)
    print("  dossiers :", dirs)
    print("  fichiers :", files[:5])  # pour ne pas spammer
    break


for cls in sorted(os.listdir(path)):
    cls_path = os.path.join(path, cls)
    if os.path.isdir(cls_path):
        imgs = os.listdir(cls_path)
        print(f"{cls} : {len(imgs)} images")


cls = sorted(os.listdir(path))[0]
cls_path = os.path.join(path, cls)
first_img_path = os.path.join(cls_path, os.listdir(cls_path)[0])

img = Image.open(first_img_path)
print("Classe :", cls)
print("Format :", img.format)
print("Taille :", img.size)
print("Mode  :", img.mode)
#img.show()

