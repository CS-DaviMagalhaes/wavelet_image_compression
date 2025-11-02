import math
import numpy as np
import cv2

# Señal original (por ejemplo las intensidades en una fila de pixeles en una imagen)
s0 = [10, 34, 20, 55, 112, 103, 41, 5, 51]

# Coeficientes del wavelet Haar
sq = 1 / math.sqrt(2)
a = [sq, sq]       # coeficientes low-pass (a_k)
b = [sq, -sq]      # coeficientes high-pass (β_k)

# ecuación 1
# Nos da la aproximación (versión suavizada de la señal)
def low_pass(signal, a):
    s1 = []
    for i in range(0, len(signal)-1, 2):
        val = a[0]*signal[i] + a[1]*signal[i+1]
        s1.append(val)
    
    return s1

# ecuación 2
# Nos da los detalles (diferencia de esquinas en la señal)
def high_pass(signal, b):
    d1 = []
    for i in range(0, len(signal)-1, 2):
        val = b[0]*signal[i] + b[1]*signal[i+1]
        d1.append(val)
    
    return d1

def dwt_2d(img):
    # aplicar 1D a cada fila
    rows_low, rows_high = [], []
    for row in img:
        rows_low.append(low_pass(row, a))
        rows_high.append(high_pass(row, b))
    rows_low, rows_high = np.array(rows_low), np.array(rows_high)

    # aplicar 1D a cada columna de la matriz que obtuvimos arriba
    A, H, V, D = [], [], [], []
    for col_low, col_high in zip(rows_low.T, rows_high.T):
        A.append(low_pass(col_low, a))
        H.append(high_pass(col_low, b))
        V.append(low_pass(col_high, a))
        D.append(high_pass(col_high, b))

    A = np.array(A).T # Aproximation (LL)
    H = np.array(H).T # Horizontal detail (HL)
    V = np.array(V).T # Vertical detail (LH)
    D = np.array(D).T # Diagonal detail (HH)
    return A, H, V, D

# cargar imagen en escala de grises
img = cv2.imread("lenna.jpg", cv2.IMREAD_GRAYSCALE)
img = img.astype(float)

# asegurar tamaño par
img = img[:img.shape[0]//2*2, :img.shape[1]//2*2]

A, H, V, D = dwt_2d(img)

print("A (LL) shape:", A.shape)
print("H (HL) shape:", H.shape)
print("V (LH) shape:", V.shape)
print("D (HH) shape:", D.shape)
