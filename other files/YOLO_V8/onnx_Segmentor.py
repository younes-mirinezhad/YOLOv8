import onnxruntime
import numpy as np
import cv2
import math
import os
import tempfile

class ONNX_Segmentor():
    def __init__(self):
        self.class_names = []
        self.colors = []
        self.session = None
        self.input_names = None
        self.input_shape = None
        self.output_names = None
        
        self.conf_threshold = 0.2
        self.iou_threshold = 0.5
        self.num_masks = 32

    def setClassName(self, className):
        self.class_names = className

        rng = np.random.default_rng(3)
        self.colors = rng.uniform(0, 255, size=(len(className), 3))

    def loadModel(self, modelPath):
        print("--- loading model")
        print("--- availble providers:", onnxruntime.get_available_providers())

        sess_options = onnxruntime.SessionOptions()

        # Enables parallel execution of operators where possible. (Default: ORT_SEQUENTIAL)
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        print(f"------> availble providers: {onnxruntime.get_available_providers()}")
        if "TensorrtExecutionProvider" in onnxruntime.get_available_providers():
            providers = ["TensorrtExecutionProvider"]
            os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"
            os.environ["ORT_TENSORRT_CACHE_PATH"] = tempfile.gettempdir()
        elif "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            providers = ["CUDAExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
            # How many threads are used inside a single operator
            sess_options.intra_op_num_threads = 4 
            # How many threads ONNX can use to run multiple operators in parallel.
            sess_options.inter_op_num_threads = 2
            # Reduces memory allocations (reuses memory buffers across runs for ops with the same input shapes.)
            sess_options.enable_mem_pattern = True
            # Reduce memory fragmentation and allocation overhead.
            sess_options.enable_cpu_mem_arena = True

        self.session = onnxruntime.InferenceSession(modelPath, sess_options=sess_options, providers=providers)

        print("------ model loaded. current providers:", providers)

        # Get model inputs info
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape

        # Get model outputs info
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

        print("------ model input shape:", self.input_shape)

    def inference(self, org_img):
        input_tensor = self.preprocess(org_img)

        # Perform inference on the image
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        return outputs
    
    def preprocess(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_shape[3], self.input_shape[2]))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def draw(self, results, image):
        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(results[0])
        self.mask_maps = self.process_mask_output(mask_pred, results[1])

        combined_img = self.draw_masks(image, draw_scores=True, mask_alpha=0.9)
        # combined_img = self.draw_masks_C(image, mask_alpha=0.9)

        return combined_img

    def process_box_output(self, box_output):

        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:4+num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], [], np.array([])

        box_predictions = predictions[..., :num_classes+4]
        mask_predictions = predictions[..., num_classes+4:]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = self.nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]

    def process_mask_output(self, mask_predictions, mask_output):

        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = self.sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(self.boxes,
                                   (self.img_height, self.img_width),
                                   (mask_height, mask_width))

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (int(self.img_width / mask_width), int(self.img_height / mask_height))
        for i in range(len(scale_boxes)):

            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))

            x1 = int(math.floor(self.boxes[i][0]))
            y1 = int(math.floor(self.boxes[i][1]))
            x2 = int(math.ceil(self.boxes[i][2]))
            y2 = int(math.ceil(self.boxes[i][3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(scale_crop_mask,
                              (x2 - x1, y2 - y1),
                              interpolation=cv2.INTER_CUBIC)

            crop_mask = cv2.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps
    
    def extract_boxes(self, box_predictions):
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes,
                                   (self.input_shape[2], self.input_shape[3]),
                                   (self.img_height, self.img_width))

        # Convert boxes to xyxy format
        boxes = self.xywh2xyxy(boxes)

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes
        
    def nms(self, boxes, scores, iou_threshold):
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]

        keep_boxes = []
        while sorted_indices.size > 0:
            # Pick the last box
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)

            # Compute IoU of the picked box with the rest
            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

            # Remove boxes with IoU over the threshold
            keep_indices = np.where(ious < iou_threshold)[0]

            # print(keep_indices.shape, sorted_indices.shape)
            sorted_indices = sorted_indices[keep_indices + 1]

        return keep_boxes

    @staticmethod
    def compute_iou(box, boxes):
        # Compute xmin, ymin, xmax, ymax for both boxes
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        # Compute intersection area
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        # Compute union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        # Compute IoU
        iou = intersection_area / union_area

        return iou

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        # Rescale boxes to original image dimensions
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])

        return boxes

    @staticmethod
    def xywh2xyxy(x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.5):
        return self.draw_detections_B(image, self.boxes, self.scores, self.class_ids, mask_alpha)

    def draw_masks(self, image, draw_scores=True, mask_alpha=0.5):
        return self.draw_detections_B(image, self.boxes, self.scores, self.class_ids, mask_alpha, mask_maps=self.mask_maps)

    def draw_detections_B(self, image, boxes, scores, class_ids, mask_alpha=0.3, mask_maps=None):
        img_height, img_width = image.shape[:2]
        size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        mask_img = self.draw_masks_B(image, boxes, class_ids, mask_alpha, mask_maps)

        # Draw bounding boxes and labels of detections
        for box, score, class_id in zip(boxes, scores, class_ids):
            color = self.colors[class_id]

            x1, y1, x2, y2 = box.astype(int)

            # Draw rectangle
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, 2)

            label = self.class_names[class_id]
            caption = f'{label} {int(score * 100)}%'
            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=size, thickness=text_thickness)
            th = int(th * 1.2)

            cv2.rectangle(mask_img, (x1, y1), (x1 + tw, y1 - th), color, -1)

            cv2.putText(mask_img, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

        return mask_img

    def draw_masks_B(self, image, boxes, class_ids, mask_alpha=0.3, mask_maps=None):
        mask_img = image.copy()

        # Draw bounding boxes and labels of detections
        for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
            color = self.colors[class_id]

            x1, y1, x2, y2 = box.astype(int)

            # Draw fill mask image
            if mask_maps is None:
                cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
            else:
                crop_mask = mask_maps[i][y1:y2, x1:x2, np.newaxis]
                crop_mask_img = mask_img[y1:y2, x1:x2]
                crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * color
                mask_img[y1:y2, x1:x2] = crop_mask_img

        return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)

    def draw_masks_C(self, org_img, mask_alpha=0.3):
        mask_overlay = org_img.copy()

        if len(self.mask_maps) > 0:
            for i, mask in enumerate(self.mask_maps):
                # Convert mask to RGB
                mask_vis = np.zeros_like(org_img)
                mask_vis[mask > 0] = self.colors[self.class_ids[i]]
                
                # Blend mask with original image
                mask_overlay = cv2.addWeighted(mask_overlay, 1, mask_vis, mask_alpha, 0)

        return mask_overlay
