from ultralytics import YOLO

class YOLO_PT_Segmentor():
    def __init__(self):
        self.model = None
        self.class_names = []

    def loadModel(self, modelPath):
        self.model = YOLO(modelPath, task='segment')
        self.model.to('cuda') # 'cuda' or 'cpu'

    def setClassName(self, className):
        self.class_names = className

    def inference(self, image):
        res = self.model(image)
        return res

    def draw(self, results, image):
        annotated_frame = results[0].plot()
        return annotated_frame