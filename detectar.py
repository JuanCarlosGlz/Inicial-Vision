import cv2
from ultralytics import YOLO
import os

# --- TRUCO DE SEGURIDAD ---
# Esta línea busca el archivo 'best.pt' más reciente dentro de la carpeta runs
ruta_modelo = "runs/detect/train2/weights/best.pt" # Ruta estándar por defecto

# Si no lo encuentra ahí, prueba con la ruta que especificamos antes
if not os.path.exists(ruta_modelo):
    ruta_modelo = "YOLO_SAM3_Runs/mi_primer_entrenamiento/weights/best.pt"

# Si sigue sin existir, lanzamos un mensaje claro
if not os.path.exists(ruta_modelo):
    print(f"ERROR: No se encontró el archivo en {ruta_modelo}")
    print("Revisa en tu carpeta 'runs' cómo se llama la carpeta de pesos.")
    exit()

# Cargar modelo
model = YOLO(ruta_modelo)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model(frame, stream=True, conf=0.5)

    for r in results:
        frame = r.plot()

    cv2.imshow("Deteccion en Vivo", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()