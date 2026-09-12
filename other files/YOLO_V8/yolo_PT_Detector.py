from ultralytics import YOLO

class YOLO_PT_Detector():
    def __init__(self):
        self.model = None
        self.class_names = []

    def loadModel(self, modelPath):
        print("--- loading model")
        self.model = YOLO(modelPath, task='detect')
        self.model.to('cuda') # 'cuda' or 'cpu'
        print("------ model loaded")

    def setClassName(self, className):
        print("--- setting class names")
        self.class_names = className

    def inference(self, image):
        res = self.model(image)
        return res

    def draw(self, results, image):
        annotated_frame = results[0].plot()
        return annotated_frame