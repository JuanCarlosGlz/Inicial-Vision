from ultralytics import YOLO
import os

if __name__ == '__main__':
    # Cargamos el modelo
    model = YOLO("yolov8n.pt")

    # Forzamos la ruta absoluta al archivo yaml
    yaml_path = os.path.abspath("data.yaml")

    model.train(
        data=yaml_path,
        epochs=25,
        imgsz=416,
        device='cpu',  # Tu procesador hará el trabajo
        workers=0
    )