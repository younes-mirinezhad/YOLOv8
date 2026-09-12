# import threading
import tensorrt as trt
import numpy as np
from cuda import cuda, cudart
import cv2
import matplotlib.pyplot as plt

# class TRT_Detector(threading.Thread):
class TRT_Detector():
    def __init__(self):
        # super().__init__()

        self.mean = None
        self.std = None
        self.n_classes = 0
        self.class_names = []

    def cuda_call(self, call):
        err, res = call[0], call[1:]
        self.check_cuda_err(err)
        if len(res) == 1:
            res = res[0]
        return res
    
    def check_cuda_err(self, err):
        if isinstance(err, cuda.CUresult):
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError("Cuda Error: {}".format(err))
        if isinstance(err, cudart.cudaError_t):
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError("Cuda Runtime Error: {}".format(err))
        else:
            raise RuntimeError("Unknown error type: {}".format(err))

    def setClassName(self, className):
        self.class_names = className

    def loadModel(self, modelPath):
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            logger.min_severity = trt.Logger.Severity.ERROR
            runtime = trt.Runtime(logger)
            trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
            with open(modelPath, "rb") as f:
                serialized_engine = f.read()
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            self.imgsz = self.engine.get_tensor_shape(self.engine.get_tensor_name(0))[2:]  # get the read shape of model, in case user input it wrong
            self.context = self.engine.create_execution_context()
            # Setup I/O bindings
            self.inputs = []
            self.outputs = []
            self.allocations = []
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                dtype = self.engine.get_tensor_dtype(name)
                shape = self.engine.get_tensor_shape(name)
                is_input = False
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    is_input = True
                if is_input:
                    self.batch_size = shape[0]
                size = np.dtype(trt.nptype(dtype)).itemsize
                for s in shape:
                    size *= s
                allocation = self.cuda_call(cudart.cudaMalloc(size))
                binding = {
                    'index': i,
                    'name': name,
                    'dtype': np.dtype(trt.nptype(dtype)),
                    'shape': list(shape),
                    'allocation': allocation,
                    'size': size
                }
                self.allocations.append(allocation)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.inputs.append(binding)
                else:
                    self.outputs.append(binding)
            return True
        except Exception as e:
            return False
    
    def inference(self, org_img):
        # img, ratio = preproc(org_img, self.imgsz, self.mean, self.std)
        img, ratio, dwdh = self.letterbox(org_img, self.imgsz)
        data = self.infer(img)

        num, final_boxes, final_scores, final_cls_inds  = data
        # final_boxes, final_scores, final_cls_inds  = data
        dwdh = np.asarray(dwdh * 2, dtype=np.float32)
        final_boxes -= dwdh
        final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
        final_scores = np.reshape(final_scores, (-1, 1))
        final_cls_inds = np.reshape(final_cls_inds, (-1, 1))
        dets = np.concatenate([np.array(final_boxes)[:int(num[0])], np.array(final_scores)[:int(num[0])], np.array(final_cls_inds)[:int(num[0])]], axis=-1)

        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            org_img = self.vis(org_img, final_boxes, final_scores, final_cls_inds, class_names=self.class_names)
        return org_img
    
    def letterbox(self, im, new_shape = (640, 640), color = (114, 114, 114), swap=(2, 0, 1)):
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # new_shape: [width, height]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
        # Compute padding [width, height]
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.transpose(swap)
        im = np.ascontiguousarray(im, dtype=np.float32) / 255.
        return im, r, (dw, dh)
    
    def rainbow_fill(size=100):  # simpler way to generate rainbow color
        cmap = plt.get_cmap('jet')
        color_list = []

        for n in range(size):
            color = cmap(n/size)
            color_list.append(color[:3])  # might need rounding? (round(x, 3) for x in color)[:3]

        return np.array(color_list)
    _COLORS = rainbow_fill(100).astype(np.float32).reshape(-1, 3)

    def vis(self, img, boxes, scores, cls_ids, class_names=None):
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            # if score < conf:
            #     continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            color = (self._COLORS[cls_id] * 255).astype(np.uint8).tolist()
            text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
            txt_color = (0, 0, 0) if np.mean(self._COLORS[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX

            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

            txt_bk_color = (self._COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(img, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])), txt_bk_color, -1)
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

        return img
    
    def infer(self, img):
        # Prepare the output data.
        outputs = []
        for shape, dtype in self.output_spec():
            outputs.append(np.zeros(shape, dtype))

        # Process I/O and execute the network.
        self.memcpy_host_to_device(self.inputs[0]['allocation'], np.ascontiguousarray(img))

        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            self.memcpy_device_to_host(outputs[o], self.outputs[o]['allocation'])
        return outputs

    # Wrapper for cudaMemcpy which infers copy size and does error checking
    def memcpy_host_to_device(self, device_ptr: int, host_arr: np.ndarray):
        nbytes = host_arr.size * host_arr.itemsize
        self.cuda_call(cudart.cudaMemcpy(device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))

    # Wrapper for cudaMemcpy which infers copy size and does error checking
    def memcpy_device_to_host(self, host_arr: np.ndarray, device_ptr: int):
        nbytes = host_arr.size * host_arr.itemsize
        self.cuda_call(cudart.cudaMemcpy(host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
    
    def output_spec(self):
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs


