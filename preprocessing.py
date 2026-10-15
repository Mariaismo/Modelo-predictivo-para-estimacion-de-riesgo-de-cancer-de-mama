### Funciones para el preprocesamiento###

import cv2
import numpy as np

def create_breast_mask(image): #conseguir máscara de la imagen (negro=fondo y blanco=mama)
    """
    Crea una máscara binaria de la mama.
    """

    image_8bit = normalize_to_uint8(image) #pasar imagen a 8 bit

    #suavizado con filtro para reducir cambios de intensidad y quitar ruido
    blurred = cv2.GaussianBlur(image_8bit, (5, 5), 0)

    #búsqueda automática de umbral de intensidad que divida los píxles en fondo/mama
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #filtro de opening para eliminar pequeños objetos
    kernel_open = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (7, 7)) #crear elipse de 7x7
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open) #aplicar el filtro a la imagen

    #filtro de closing para eliminar huecos
    kernel_close = cv2.getStructuringElement(  cv2.MORPH_ELLIPSE, (31, 31))#crear elipse de 31x31
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)#aplicar el filtro a la imagen

    return mask

def get_largest_component(mask):
    """
    Se queda únicamente con el componente conectado de mayor tamaño.
    """

    num_labels, labels, stats, _ = ( cv2.connectedComponentsWithStats( mask, connectivity=8)) #opencv busca grupos de píxeles conectados
    #num_labels es nº de componentes encontrados, labels es una imagen en la que cada pixel indica a qué componente pretenece, stats tiene info de cada componente (posición xy, área, dimensiones...)

    if num_labels <= 1: #número de componentes encontrados
        return np.zeros_like(mask)
    areas = stats[1:, cv2.CC_STAT_AREA] #el label 0 es el fondo y se ignora 
    largest_label = 1 + np.argmax(areas) #seleccionar el índice del componenete con mayor área
    largest_mask = np.zeros_like(mask) #imagen negra
    largest_mask[labels == largest_label] = 255 #se pinta de blanco la zona del componente seleccionado

    return largest_mask

def get_bounding_box(mask):
    """
    Obtiene el bounding box del componente de la mama.
    """

    contours, _ = cv2.findContours( mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #buscar el contorno de la mama
    if len(contours) == 0:
        return None
    largest_contour = max(contours, key=cv2.contourArea) #seleccionar el contorno de mayor área
    x, y, w, h = cv2.boundingRect(largest_contour) #obtener posición xy, ancho y alto

    return x, y, w, h

def crop_breast(image):
    """
    Recorta la región de la mama manteniendo la imagen original.
    Devuelve:
        cropped: imagen original recortada
        cropped_mask: máscara correspondiente al mismo recorte
    """

    # Recorte vertical inicial para quitar bordes de la imagen
    top_crop = 100
    bottom_crop = 100
    image_preprocessed = image[top_crop:-bottom_crop, :]

    # Crear máscara de la mama
    breast_mask = create_breast_mask(image_preprocessed)

    # Obtener componente más grande
    largest_mask = get_largest_component(breast_mask)

    # Obtener bounding box
    x, y, w, h = get_bounding_box(largest_mask)

    print("\nBounding box:")
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"w = {w}")
    print(f"h = {h}")

    # Añadir margen de seguridad
    margin_x = 300
    margin_y = 300

    x_min = max(0, x - margin_x)
    x_max = min(image_preprocessed.shape[1], x + w + margin_x)
    y_min = max(0, y - margin_y)
    y_max = min(image_preprocessed.shape[0], y + h + margin_y)

    # Recortar la mamografía original
    cropped = image_preprocessed[y_min:y_max, x_min:x_max].copy()

    # Recortar la MÁSCARA con las mismas coordenadas
    cropped_mask = largest_mask[y_min:y_max, x_min:x_max].copy()

    # Eliminar pequeño borde blanco
    border_remove = 10
    if x < image_preprocessed.shape[1] / 2:
        # Mama izquierda
        cropped = cropped[:, border_remove:]
        cropped_mask = cropped_mask[:, border_remove:]

    else:
        # Mama derecha
        cropped = cropped[:, :-border_remove]
        cropped_mask = cropped_mask[:, :-border_remove]

    # Comprobación
    assert cropped.shape == cropped_mask.shape, (
        f"Imagen y máscara tienen tamaños diferentes: "
        f"{cropped.shape} vs {cropped_mask.shape}"
    )

    print("Tamaño original:", image.shape)
    print("Tamaño después del recorte inicial:", image_preprocessed.shape)
    print("Tamaño final:", cropped.shape)
    print("Tamaño máscara:", cropped_mask.shape)

    return cropped, cropped_mask

def resize_with_padding(image, target_size=(512, 512)):
    """
    Redimensiona una imagen manteniendo su proporción y añade padding negro hasta alcanzar target_size. Devuelve una imagen redimensionada con padding.
    Args:
        image: imagen de entrada.
        target_size: tamaño final (alto, ancho).
    """

    target_height, target_width = target_size
    height, width = image.shape

    # calcular factor de escala: cuánto se puede reducir sin perder la proporción
    scale = min(target_width / width, target_height / height)

    # calcular nuevo tamaño manteniendo la proporción
    new_width = int(width * scale)
    new_height = int(height * scale)

    # redimensionar
    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # crear imagen negra con dimensiones finales
    padded = np.zeros((target_height, target_width), dtype=image.dtype)

    # calcular dónde colocar la imagen centrada
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2

    # insertar imagen redimensionada en imagen negra
    padded[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

    return padded

def normalize_image(image, mask, lower_percentile=1, upper_percentile=99):
    """
    Normaliza la imagen usando los percentiles calculados únicamente sobre la región de la mama.
    """

    # Píxeles que pertenecen a la mama
    breast_pixels = image[mask > 0]

    # Percentiles de intensidad de la mama para quedarnos con los pixeles que están dentro de ese rango
    lower = np.percentile(breast_pixels, lower_percentile)
    upper = np.percentile(breast_pixels, upper_percentile)

    # Evitar división por cero
    if upper <= lower:
        return np.zeros_like(image, dtype=np.float32)

    # Limitar intensidades al rango P1-P99: los valores que quedan por fuera de los percentiles se truncan a los percentiles
    image_clipped = np.clip(image, lower, upper)

    # Normalizar a [0, 1]
    normalized = (image_clipped - lower) / (upper - lower)

    return normalized.astype(np.float32)

def find_binary_roi(roi_case_dir, image_shape):
    """
    Busca la máscara ROI binaria.
    La máscara normalmente es un DICOM 0/255 con las mismas dimensiones que la mamografía.
    Si las dimensiones no coinciden, se redimensiona usando INTER_NEAREST para mantener la máscara binaria.
    """
    for roi_file in roi_case_dir.rglob("*.dcm"):
        ds = pydicom.dcmread(roi_file)
        mask = ds.pixel_array
        del ds

        unique_values = np.unique(mask)

        # Comprobar que es una máscara binaria
        if not (mask.ndim == 2 and set(unique_values).issubset({0, 255})):
            del mask
            continue

        # Si las dimensiones no coinciden, ajustar
        if mask.shape != image_shape:
            print(f"AVISO: redimensionando ROI " f"{mask.shape} -> {image_shape}")
            mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)

        return roi_file, mask

    return None, None