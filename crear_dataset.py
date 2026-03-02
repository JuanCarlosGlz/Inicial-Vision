import cv2
import os
import time

# 1. Configuración de carpetas
NOMBRE_PATRON = "mi_objeto"  # Cambia esto por el nombre de lo que detectas
CARPETA_DESTINO = os.path.join("datasets", "capturas")

if not os.path.exists(CARPETA_DESTINO):
    os.makedirs(CARPETA_DESTINO)
    print(f"Carpeta creada: {CARPETA_DESTINO}")

# 2. Iniciar Cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
count = 0

print("--- MODO RECOLECTOR ---")
print("1. Mueve el objeto en diferentes angulos y distancias.")
print("2. Presiona 'S' para capturar.")
print("3. Presiona 'Q' para terminar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Copia para mostrar info en pantalla sin manchar la foto original
    img_display = frame.copy()
    cv2.putText(img_display, f"Capturas: {count}", (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Capturando Patrones", img_display)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
        # Guardar la imagen original (limpia, sin texto)
        timestamp = int(time.time() * 100)
        nombre_archivo = f"{NOMBRE_PATRON}_{timestamp}.jpg"
        ruta_completa = os.path.join(CARPETA_DESTINO, nombre_archivo)
        
        cv2.imwrite(ruta_completa, frame)
        count += 1
        print(f"[{count}] Guardada: {nombre_archivo}")
        
        # Flash visual para saber que capturó
        cv2.rectangle(img_display, (0,0), (frame.shape[1], frame.shape[2]), (255,255,255), -1)
        cv2.imshow("Capturando Patrones", img_display)
        cv2.waitKey(100) 

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nProceso terminado. Tienes {count} fotos en {CARPETA_DESTINO}")