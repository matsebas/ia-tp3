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


def test_dummy_plot():
    # Crear una matriz binaria simple para prueba
    dummy_image = np.array([
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    ])
    display_image(dummy_image, title="Dummy Image")


def test_hopfield():
    # Crear imágenes de referencia y de prueba
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

    test_image_correct = np.array([
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

    test_image_incorrect = np.array([
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    ])

    # Preprocesar imágenes (convertir a binario y aplanar)
    preprocessed_reference = preprocess_image(reference_image)
    preprocessed_test_correct = preprocess_image(test_image_correct)
    preprocessed_test_incorrect = preprocess_image(test_image_incorrect)

    # Crear y entrenar la red de Hopfield usando Hebbian Learning con la imagen de referencia
    hopfield_net = HopfieldNetwork(n_neurons=100)
    hopfield_net.train([preprocessed_reference])

    # Recuperar el patrón para la imagen correcta
    recalled_image_correct = hopfield_net.recall(preprocessed_test_correct).reshape(10, 10)
    center_correct = find_ring_center(recalled_image_correct)

    # Recuperar el patrón para la imagen incorrecta
    recalled_image_incorrect = hopfield_net.recall(preprocessed_test_incorrect).reshape(10, 10)
    center_incorrect = find_ring_center(recalled_image_incorrect)

    # Visualizar las imágenes recuperadas y mostrar las coordenadas del centro del aro
    display_image(recalled_image_correct, title=f"Imagen Recuperada Correcta (Centro: {center_correct})")
    display_image(recalled_image_incorrect, title=f"Imagen Recuperada Incorrecta (Centro: {center_incorrect})")

    # Verificar que los centros son diferentes
    print(f"Centro correcto: {center_correct}")
    print(f"Centro incorrecto: {center_incorrect}")


if __name__ == '__main__':
    test_dummy_plot()
    test_hopfield()
