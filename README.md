### Implementación del Modelo de Hopfield para Identificación de Imágenes

Este repositorio contiene una implementación de un modelo de red neuronal de Hopfield en Python, utilizado para identificar y recuperar patrones en imágenes en el contexto de un caso de estudio específico.

### Autor
- Matias Sebastiao
- Email: matisebastiao@gmail.com

### Materia
- Inteligencia Artificial

### Año
- 2024

## Contenido del repositorio

### Archivos principales
1. `main.py`: Contiene la implementación del modelo de Hopfield, funciones de preprocesamiento y visualización de imágenes, y pruebas del prototipo.

### Ejecución
Para ejecutar el prototipo de identificación de imágenes utilizando el modelo de Hopfield, simplemente ejecute el archivo `main.py` desde la línea de comandos o desde su IDE preferido. Asegúrese de tener Python y las dependencias necesarias instaladas en su sistema.

### Dependencias
Asegúrese de tener instaladas las siguientes dependencias antes de ejecutar el código:
```bash
pip install numpy
pip install matplotlib
pip install opencv-python
```

### Cálculo de la Distancia en el Algoritmo

El algoritmo de identificación de imágenes utilizando el modelo de Hopfield incluye un cálculo de la distancia entre los patrones recuperados y los patrones de referencia para evaluar la precisión del modelo. Se utilizan las siguientes métricas de distancia:

1. **Distancia Euclidiana**:
    - Calcula la distancia directa entre dos puntos en un espacio bidimensional, que representan los centros de los patrones en las imágenes.
    - Fórmula: $`\( \text{Distancia Euclidiana} = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2} \)`$
    - Esta métrica proporciona una medida de cuán lejos están los centros de los patrones recuperados de los centros de los patrones de referencia.

2. **Distancia en X y en Y**:
    - Calcula la distancia en los ejes X e Y entre los centros de los patrones.
    - Fórmulas:
        - $`\( \text{Distancia en X} = x_2 - x_1 \)`$
        - $`\( \text{Distancia en Y} = y_2 - y_1 \)`$
    - Estas métricas proporcionan una medida más específica de cuánto se deben desplazar los centros de los patrones recuperados en cada eje para coincidir con los centros de los patrones de referencia.

### Ejemplo de Ejecución

Al ejecutar el archivo `main.py`, el algoritmo entrenará la red de Hopfield con varios patrones desplazados y luego recuperará patrones desde imágenes de prueba. Los resultados incluirán la visualización de las imágenes recuperadas y la impresión de las coordenadas del centro, junto con las distancias calculadas en X, Y y euclidiana.

```python
Test 0: Centro recuperado: (4.5, 4.5), Centro de referencia: (4.5, 4.5), Distancia euclidiana: 0.0, Desplazamiento en X: 0.0, Desplazamiento en Y: 0.0
Test 1: Centro recuperado: (5.5, 5.5), Centro de referencia: (4.5, 4.5), Distancia euclidiana: 1.4142135623730951, Desplazamiento en X: 1.0, Desplazamiento en Y: 1.0
Test 2: Centro recuperado: (3.5, 3.5), Centro de referencia: (4.5, 4.5), Distancia euclidiana: 1.4142135623730951, Desplazamiento en X: -1.0, Desplazamiento en Y: -1.0
Test 3: Centro recuperado: (6.5, 4.5), Centro de referencia: (4.5, 4.5), Distancia euclidiana: 2.0, Desplazamiento en X: 2.0, Desplazamiento en Y: 0.0
Test 4: Centro recuperado: (4.5, 2.5), Centro de referencia: (4.5, 4.5), Distancia euclidiana: 2.0, Desplazamiento en X: 0.0, Desplazamiento en Y: -2.0
```

### Resumen

Este repositorio proporciona una implementación completa del modelo de Hopfield para la identificación de imágenes, incluyendo el cálculo de distancias para evaluar la precisión del modelo. La utilización de distancias euclidianas y en los ejes X e Y permite una evaluación detallada de los resultados.


$E = mc^2$