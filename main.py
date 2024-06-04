## TP3 Prototipo de implementación del modelo de Hopfield
## autor: Matias Sebastiao
## email: matisebastiao@gmail.com
## materia: Inteligencia Artificial
## año: 2024

import numpy as np
import matplotlib.pyplot as plt
import cv2


class HopfieldNetwork:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons  # Número de neuronas en la red
        self.weights = np.zeros((n_neurons, n_neurons))  # Inicialización de la matriz de pesos

    # Método de entrenamiento utilizando la regla de aprendizaje de Hebb
    def train(self, patterns):
        for p in patterns:
            p = p.reshape(-1, 1)  # Convertir el patrón en un vector columna
            self.weights += np.dot(p, p.T)  # Actualizar los pesos según la regla de Hebb
        self.weights[np.diag_indices(self.n_neurons)] = 0  # Asegurarse de que no haya auto-conexiones

    # Método para recordar patrones a partir de una entrada dada
    def recall(self, pattern, steps=10):
        pattern = pattern.copy()
        for _ in range(steps):
            for i in range(self.n_neurons):
                activation = np.dot(self.weights[i], pattern)
                pattern[i] = 1 if activation >= 0 else -1
        return pattern


# Función para preprocesar la imagen
def preprocess_image(image):
    image_uint8 = (image * 255).astype(np.uint8)
    _, binary_image = cv2.threshold(image_uint8, 127, 255, cv2.THRESH_BINARY)
    binary_image = binary_image / 255  # Escalar los valores de vuelta a 0 y 1
    binary_image[binary_image == 0] = -1  # Convertir los ceros a -1
    return binary_image.flatten()


# Función para visualizar la imagen
def display_image(image, title="Image"):
    plt.imshow(image.reshape((10, 10)), cmap='gray', vmin=-1, vmax=1)
    plt.title(title)
    plt.axis('off')
    plt.show()


# Función para encontrar la posición del centro del aro
def find_ring_center(image):
    coordinates = np.argwhere(image == 1)
    if len(coordinates) == 0:
        return None
    x_center = np.mean(coordinates[:, 0])
    y_center = np.mean(coordinates[:, 1])
    return (x_center, y_center)


# Función para desplazar el aro en la imagen
def shift_ring(image, shift_x, shift_y):
    shifted_image = np.roll(image, shift_x, axis=0)
    shifted_image = np.roll(shifted_image, shift_y, axis=1)
    return shifted_image


# Función para calcular la distancia euclidiana entre dos centros
def euclidean_distance(center1, center2):
    return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)


# Función para calcular la distancia en X y en Y entre dos centros
def xy_distance(center1, center2):
    return (center1[0] - center2[0], center1[1] - center2[1])


# Función para calcular la distancia entre dos patrones
def pattern_distance(pattern1, pattern2):
    return np.sum(np.abs(pattern1 - pattern2))


def test_hopfield():
    # Crear la imagen de referencia
    reference_image = np.array([
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    ])

    # Generar imágenes desplazadas para entrenamiento
    training_images = [reference_image]
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for shift in shifts:
        training_images.append(shift_ring(reference_image, shift[0], shift[1]))

    # Preprocesar imágenes de entrenamiento
    preprocessed_training_images = [preprocess_image(img) for img in training_images]

    # Crear la red de Hopfield y entrenarla con las imágenes desplazadas
    hopfield_net = HopfieldNetwork(n_neurons=100)
    hopfield_net.train(preprocessed_training_images)

    # Imágenes de prueba
    test_images = [
        reference_image,
        shift_ring(reference_image, 1, 1),
        shift_ring(reference_image, -1, -1),
        shift_ring(reference_image, 2, 0),
        shift_ring(reference_image, 0, -2),
    ]

    for i, test_image in enumerate(test_images):
        preprocessed_test = preprocess_image(test_image)
        recalled_image = hopfield_net.recall(preprocessed_test).reshape(10, 10)
        center_recalled = find_ring_center(recalled_image)
        center_reference = find_ring_center(reference_image)
        distance = euclidean_distance(center_recalled, center_reference)
        x_distance, y_distance = xy_distance(center_recalled, center_reference)

        # Visualizar la imagen recuperada y mostrar las coordenadas del centro del aro
        display_image(recalled_image, title=f"Imagen Recuperada {i} (Centro: {center_recalled})")
        print(
            f"Test {i}: Centro recuperado: {center_recalled}, Centro de referencia: {center_reference}, Distancia euclidiana: {distance}, Desplazamiento en X: {x_distance}, Desplazamiento en Y: {y_distance}")


if __name__ == '__main__':
    test_hopfield()
