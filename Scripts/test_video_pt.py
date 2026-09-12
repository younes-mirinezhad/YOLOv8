class_names = ["Gap"]

if __name__ == "__main__":
    isVideo = True
    resize = True
    width = height = 720
    filePath = "/mnt/HDD_1/Work/Companies/Petanux/GeneralStockWarning/Dataset/2_sample_data/videos/vlc-record-2026-05-05-12h22m34s-rtsp___10.18.18.36_8556_video0-.mp4"
    modelPath = "/mnt/HDD_1/Work/Projects/YOLO/Python/Train/Petanux/ShelfGapsSegmentation/runs/segmentation/V0.1_26n/weights/best.pt"

    # from yolo_PT_Detector import YOLO_PT_Detector
    # model = YOLO_PT_Detector()

    from yolo_PT_Segmentor import YOLO_PT_Segmentor
    model = YOLO_PT_Segmentor()

    # from yolo_onnx_Segmentor import ONNX_Segmentor
    # model = ONNX_Segmentor()

    model.setClassName(class_names)
    model.loadModel(modelPath)

    import cv2

    if isVideo:
        cap = cv2.VideoCapture(filePath)
        fnum = 0

        sumTime = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            start_time = cv2.getTickCount()
            result = model.inference(frame)
            end_time = cv2.getTickCount()
            inference_time = (end_time - start_time) / cv2.getTickFrequency() * 1000
            sumTime += inference_time

            anotatedImage = model.draw(result, frame)

            if resize:
                frame_height, frame_width = anotatedImage.shape[:2]
                frame_aspect = frame_width / frame_height
                window_aspect = width / height
                if frame_aspect > window_aspect:
                    scale = width / frame_width
                else:
                    scale = height / frame_height
                new_width = int(frame_width * scale)
                new_height = int(frame_height * scale)
                anotatedImage = cv2.resize(
                    anotatedImage, (new_width, new_height), interpolation=cv2.INTER_AREA
                )

            cv2.imshow("Inference results", anotatedImage)

            fnum += 1
            print(
                f"-----> Frame: {fnum} , Inference time: {inference_time:.2f} ms --- average: {sumTime / fnum:.2f} ms"
            )
            key = cv2.waitKey(1)
            if key == ord("q"):
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
