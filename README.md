# CoursComputerVision
Cours Artefact de vision par ordinateur

# 1 Le repo Git. 

Un repo Git propre doit comporter :
- un README clair et, si possible, illustré. 
- un .gitignore afin de ne pas surcharger le repo
- un requirements.txt afin que l'utilisateur puisse lancer un `pip install -r requirements.txt`. 
- un dossier ou plusieurs dossiers comportant les différents modules. 
- un dossier ou plusieurs avec les test unitaires propres à chacun des modules. 

L'utilisateur doit pouvoir lancer directement le programme sur environnement virtuel par `python3 -m nomDuProgramme.py`, on utilise donc des argparses dans les modules qui contiennent tous des classes (Programmation Orienté Objet). 

Ce cours permet de comprendre le code afin d'utiliser les modules Python de vision par ordinateur. 

# 2 IA sans deep learning, utilisation d'InsightFace. 

Le programme de détection de visage permet de détecter les visages et de créer un modèle d'IA capable de distinguer une personne parmis ces visages. 

On a donc un module pour la détection de personnes et un module pour la reconnaissance de personnes. 

Pour reconnaitre les individus, il faut d'abord détecter leurs visages. On utilisera ici InsightFace pour cela. Le module `detectFace.py` prend une image en entrée, reconnait les visage et calcule les embeddings. 

Enfin pour distinguer une personne, on utisera un Support Vector Classifier. Les embeddings sont une vectorisation de l'image, le SVC génère une prédiction sans nécessiter de Deep Learning.

# 3 IA avec deep learning, entrainement de Yolo. 

Ce dossier permet d'entrainer Yolo sur la segmentation d'image et/ou les rectangles de détection. On utilisera ici la version `segmentée` de Yolo

Pour l'instant pas développé.

# 3 Installation. 

`git clone git@github.com:GeorgesSchmidt/ComputerVision.git`. 

`cd ComputerVision` 

`python3 -m venv .venv` 

`source .venv/bin/activate` 

`pip install -r requirements.txt` 

`python3 -m Faces.extractFrames ./Videos/kingCharles.mp4` 

Pour l'instant le module n'a fait que lire la vidéo. 

`python3 -m Faces.createModel new_model.pt` 

on peut remplacer `new_model.pt` par un autre nom de modèle tant que l'on garde l'extension `.pt`. 

Pour créer le film avec la détection. 

`python3 -m Faces.extractFrames ./Videos/kingCharles.mp4 ./Videos/selectedDetection.mp4 new_model.pt` 











