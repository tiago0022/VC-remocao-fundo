import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt

n_colunas = 2
n_linhas = 1

global indice
indice = 0


def exibe(imagem, nome=''):
    global indice
    indice = indice + 1
    plt.subplot(n_linhas, n_colunas, indice), plt.imshow(imagem, cmap='gray')
    plt.title(nome), plt.xticks([]), plt.yticks([])


caminho_imagem = 'in/blender.png'

if len(sys.argv) >= 2:
    caminho_imagem = sys.argv[1]


# == Parameters
BLUR = 21
CANNY_THRESH_1 = 10
CANNY_THRESH_2 = 100
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (1.0, 1.0, 1.0)  # In BGR format


# -- Read image
img = cv2.imread(caminho_imagem)
img_rgb = img.copy()
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

exibe(img_rgb, 'Original')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -- Edge detection
edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
edges = cv2.dilate(edges, None)
edges = cv2.erode(edges, None)

# -- Find contours in edges, sort by area
contour_info = []
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
for c in contours:
    contour_info.append((
        c,
        cv2.isContourConvex(c),
        cv2.contourArea(c),
    ))
contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
max_contour = contour_info[0]

# -- Create empty mask, draw filled polygon on it corresponding to largest contour ----
# Mask is black, polygon is white
mask = np.zeros(edges.shape)
cv2.fillConvexPoly(mask, max_contour[0], (255))

# -- Smooth mask, then blur it
mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

# -- Blend masked img into MASK_COLOR background
mask_stack = mask_stack.astype('float32') / 255.0
img = img.astype('float32') / 255.0
masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR)
masked = (masked * 255).astype('uint8')

rgba = cv2.cvtColor(masked, cv2.COLOR_RGB2BGRA)

for i in range(len(rgba)):
    for j in range(len(rgba[0])):
        if (rgba[i][j] == [0, 0, 255, 255]).all():
            rgba[i][j] = [0, 0, 0, 0]

exibe(rgba, 'Resultado')

caminho_saida = caminho_imagem.replace('in/', 'out/')

cv2.imwrite(f'{caminho_saida}.png', rgba)

plt.show()
