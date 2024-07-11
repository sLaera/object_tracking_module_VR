import cv2

# Carica l'immagine
image = cv2.imread('a.png')

# Dimensioni originali dell'immagine
original_height, original_width = image.shape[:2]

# Ridimensiona l'immagine al 89%
scale_percent = 89
width = int(original_width * scale_percent / 100)
height = int(original_height * scale_percent / 100)
dim = (width, height)

resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Dimensioni dell'immagine ridimensionata
resized_height, resized_width = resized_image.shape[:2]

# Dimensioni del ritaglio
crop_width = 600
crop_height = 450

# Calcola le coordinate per ritagliare il centro
start_x = resized_width // 2 - crop_width // 2
start_y = resized_height // 2 - crop_height // 2

# Effettua il ritaglio
cropped_image = resized_image[start_y:start_y + crop_height, start_x:start_x + crop_width]

# Salva l'immagine risultante
cv2.imwrite('cropped_image.png', cropped_image)

# Mostra l'immagine risultante (facoltativo)
cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()