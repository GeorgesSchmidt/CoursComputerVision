import cv2

# Chemin vers le fichier vidéo
video_path = '/Users/georgesschmidt/VisualCodeProjects/CoursComputerVision/Videos/selectedDetection.mp4'

# Ouvrir la vidéo
cap = cv2.VideoCapture(video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Nombre total de frames : {total_frames}")
# Vérifier si la vidéo s'est ouverte correctement
if not cap.isOpened():
    print("Erreur lors de l'ouverture du fichier vidéo.")
    exit()

# Définir le numéro de l'image que vous souhaitez extraire
frame_number = 50

# Déplacer le lecteur à l'image désirée
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

# Lire l'image
ret, frame = cap.read()

# Vérifier si l'image a été lue correctement
if ret:
    cv2.imwrite('picKing.png', frame)
    # Afficher l'image
    cv2.imshow('Image', frame)
    cv2.waitKey(0)  # Attendre une touche pour fermer la fenêtre
else:
    print("Erreur lors de la lecture de l'image.")

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
