import cv2
import numpy as np

# Inicializa el Filtro de Kalman
kalman = cv2.KalmanFilter(4, 2)

# Definir la matriz de transición de estado (A), que relaciona el estado anterior con el actual
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)

# La matriz de medida (H), que transforma el estado real al espacio de medidas
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)

# Matriz de covarianza del proceso (Q), define la incertidumbre en la evolución del estado
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

# Matriz de covarianza de medida (R), define la incertidumbre en las mediciones
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

# Estado inicial (x, y, velocidad_x, velocidad_y)
state = np.array([0, 0, 0, 0], np.float32)

# Medida inicial (x, y)
measurement = np.array([0, 0], np.float32)

# Crear una ventana para mostrar
cv2.namedWindow('Kalman Tracker')

# Simular el seguimiento de un punto
while True:
    # Simula un movimiento aleatorio del objeto a seguir
    state[0] += np.random.randn() * 2
    state[1] += np.random.randn() * 2

    # Actualiza la predicción de Kalman
    prediction = kalman.predict()
    pred_x, pred_y = int(prediction[0]), int(prediction[1])

    # Generar medición con ruido (para simular mediciones reales)
    measurement[0] = state[0] + np.random.randn() * 0.1
    measurement[1] = state[1] + np.random.randn() * 0.1

    # Actualiza el filtro de Kalman con la medida
    kalman.correct(measurement)

    # Dibuja el resultado
    img = np.zeros((500, 500, 3), np.uint8)
    cv2.circle(img, (int(measurement[0]) + 250, int(measurement[1]) + 250), 5, (0, 255, 0), -1)  # Punto medido
    cv2.circle(img, (pred_x + 250, pred_y + 250), 5, (0, 0, 255), -1)  # Punto predicho

    # Mostrar la imagen
    cv2.imshow('Kalman Tracker', img)

    if cv2.waitKey(30) & 0xFF == 27:  # Presiona 'ESC' para salir
        break

cv2.destroyAllWindows()
