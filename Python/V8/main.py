class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
               'hair drier', 'toothbrush' ]

if __name__ == '__main__':
    isVideo = True
    filePath = "Path/To/File"
    modelPath = "Path/To/Model"

    # ONNX
    # from ultralytics import YOLO
    # model = YOLO(modelPath)
    # model.export(format="onnx", batch=1, opset=12, imgsz=640) #simplify=True, dynamic=False

    # TensorRT Engine
    # from v1_EngineBuilder import Builder
    # builder = Builder()
    # modelPath_PT = "files/YOLOv8n.pt"
    # modelPath_ONNX = "files/YOLOv8n_b1_640_e2e.onnx"
    # modelPath_Engine = "files/YOLOv8n_b1_640_e2e.engine"
    # builder.setModelPath(modelPath_PT, modelPath_ONNX, modelPath_Engine)
    # builder.setConfigs(input_shape=[1, 3, 640, 640], topk=100, conf_thres=0.2, iou_thres=0.5)
    # builder.build()

    # modelPath = "files/YOLOv8n.pt"
    # from yolo_PT_Detector import YOLO_PT_Detector
    # model = YOLO_PT_Detector()

    # modelPath = "files/YOLOv8n_b1_640_e2e.engine"
    # from v1_TRT_Detector import TRT_Detector
    # model = TRT_Detector()

    # modelPath = "files/YOLOv8n_seg.pt"
    # from yolo_PT_Segmentor import YOLO_PT_Segmentor
    # model = YOLO_PT_Segmentor()

    modelPath = "files/yolov8n-seg_b1_640.onnx"
    from onnx_Segmentor import ONNX_Segmentor
    model = ONNX_Segmentor()

    model.setClassName(class_names)
    model.loadModel(modelPath)
    filePath = "files/11.mp4"

    import cv2
    if isVideo:
        cap = cv2.VideoCapture(filePath)
        fnum = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            start_time = cv2.getTickCount()
            result = model.inference(frame)
            end_time = cv2.getTickCount()
            inference_time = (end_time - start_time) / cv2.getTickFrequency() * 1000

            anotatedImage = model.draw(result, frame)
            cv2.imshow("Inference results", anotatedImage)
            
            fnum += 1
            print(f"-----> Frame: {fnum} , Inference time: {inference_time:.2f} ms")
            key = cv2.waitKey()
            if key == ord('q'):
                break
            else:
                continue
            
        cap.release()
        cv2.destroyAllWindows()
    else:
        frame = cv2.imread(filePath)

        trt_annotated = model.inference(frame)
        cv2.imshow("TRT Inference", trt_annotated)

        key = cv2.waitKey()

        cv2.destroyAllWindows()
