import cv2
from ultralytics import YOLO

# --- CONFIGURACIÓN DE ESCALA ---
# Mide con una regla en tu pantalla: ¿Cuántos píxeles hay en 10cm reales?
# Ejemplo: si 100 píxeles = 10cm, entonces PPM = 10.0
PIXELS_PER_CM = 15.5  # AJUSTA ESTE VALOR SEGÚN TU DISTANCIA

model = YOLO("runs/detect/train3/weights/best.pt")
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Obtener dimensiones del frame para hallar el centro
    alto, ancho, _ = frame.shape
    centro_pantalla_x = ancho // 2
    centro_pantalla_y = alto // 2

    # Dibujar los ejes del "Mundo Real" (Opcional, para visualizar)
    cv2.line(frame, (centro_pantalla_x, 0), (centro_pantalla_x, alto), (255, 255, 255), 1)
    cv2.line(frame, (0, centro_pantalla_y), (ancho, centro_pantalla_y), (255, 255, 255), 1)
    cv2.circle(frame, (centro_pantalla_x, centro_pantalla_y), 5, (0, 0, 255), -1)

    results = model(frame, stream=True, conf=0.6)

    for r in results:
        for box in r.boxes:
            # 1. Obtener centro del objeto en PÍXELES
            x1, y1, x2, y2 = box.xyxy[0]
            obj_x = int((x1 + x2) / 2)
            obj_y = int((y1 + y2) / 2)

            # 2. Calcular distancia en PÍXELES desde el centro (0,0)
            # Restamos para que el centro de la cámara sea el origen
            dist_pixeles_x = obj_x - centro_pantalla_x
            dist_pixeles_y = centro_pantalla_y - obj_y # Invertimos Y para que arriba sea positivo

            # 3. CONVERSIÓN A MUNDO REAL (cm)
            mundo_x = dist_pixeles_x / PIXELS_PER_CM
            mundo_y = dist_pixeles_y / PIXELS_PER_CM

            # Visualización
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (obj_x, obj_y), 5, (255, 0, 0), -1)
            
            # Mostrar coordenadas reales
            texto = f"X: {mundo_x:.1f}cm, Y: {mundo_y:.1f}cm"
            cv2.putText(frame, texto, (obj_x + 10, obj_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Coordenadas del Mundo Real (Origen en Centro)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()