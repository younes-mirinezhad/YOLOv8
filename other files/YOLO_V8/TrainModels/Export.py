from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("/media/HDD_1/Work/Projects/Watchtower/02_Flare/Development/Models/Segmentation/V2_AryaSasol/n_InAug_1/best.pt")
    model.export(format="onnx", batch=1, opset=12, imgsz=640) #simplify=True, dynamic=False

