import cv2

# Probamos los índices más comunes para cámaras externas
for i in [1, 2, 0, -1]:
    print(f"Probando cámara en índice {i}...")
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ ¡ÉXITO! Cámara detectada en el índice {i}")
            cv2.imshow(f"Prueba Indice {i}", frame)
            cv2.waitKey(2000) # Muestra la cámara 2 segundos
            cap.release()
            cv2.destroyAllWindows()
            break
        cap.release()
    else:
        print(f"❌ Índice {i} no disponible.")