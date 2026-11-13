import sys
import math

import numpy as np
from PIL import Image


# Load da imagem e converter num array numpy
# PARAMS: path da imagem
# Return: array da imagem em int64 e valor maximo possivel de um pixel
def load_image(path):

    img = Image.open(path)

    if img.mode == 'P': # imagem que guarda indices
        img = img.convert('RGB')

    # valor maximo de um pixel conforme a profundidade
    max_val = 65535 if img.mode in ("I;16", "I;16B", "I;16L", "I") else 255

    arr = np.asarray(img, dtype=np.uint64)

    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    return arr, max_val



# Calcula o erro absoluto maximo por pixel/canal
# PARAMS: imagem original e imagem a comparar ( com as mesmas dimensoes da imagem 1)
# Return: maior valor do valor absoluto (original - modified) em todos os pixeis e canais
def max_abs_error(original, modified):

    dif = np.abs(original - modified)

    return int(dif.max())

# Calcula o erro quantico médio entre duas imagens
# PARAMS: imagem original e imagem a comparar ( com as mesmas dimensoes da imagem 1)
# Return: média de (original - modified) ** 2
def mse(original, modified):

    sq = (original - modified).astype(np.float64) ** 2

    return float(sq.mean())

# Calcula o PSNR
# PARAMS: imagem original e imagem a comparar ( com as mesmas dimensoes da imagem 1), valor maximo possivel de um pixel 255
# Return: PSNR ou math.inf caso sejam a mesma imagem
def psnr(original, modified, max_val=255):

    error = mse(original, modified)

    if error == 0:
        return math.inf

    return 10 * math.log10((max_val ** 2) / error)


def main():

    # Verficar os argumentos
    if len(sys.argv) != 3:
        print("Usage: python img_cmp.py input_image output_image")
        return 1

    orig_path, modified_path = sys.argv[1], sys.argv[2]

    # Carregar as imagens
    try:
        original, max_val = load_image(orig_path)
        modified, _ = load_image(modified_path)
    except (OSError, ValueError) as e:
        print(f"Erro ao abrir a imagem {e}")
        return 1

    # verificar se tem as mesmo tamanho e canais
    if original.shape != modified.shape:
        print(f"Erro: dimensoes diferentes")
        return 1

    max_err = max_abs_error(original, modified)
    value = psnr(original, modified, max_val)

    # Mostrar resultados
    print(f"Max error: {max_err}")
    if math.isinf(value):
        print("PSNR: imagens identicas")
    else:
        print(f"PSNR: {value}")

    return 0

if __name__ == "__main__":
    sys.exit(main())