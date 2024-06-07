import torch
import torch.nn as nn
from torch import Tensor, Graph, Value
from typing import Tuple
from ultralytics import YOLO
from io import BytesIO
import onnx
try: 
    import onnxsim 
except ImportError: 
    onnxsim = None

def main():
    weight_pt = "path/to/model.pt" # PyTorch yolov8 weights
    ONNX_Path = weight_pt.replace(".pt", "_end2end.onnx") # ONNX yolov8 weights
    device = "cuda:0" # cpu , cuda:0 --> Export device
    input_shape = [1, 3, 640, 640] # Model input shape only for api builder
    opset = 11 # ONNX opset version
    topk = 100 # Max number of detection bboxes
    sim = True # simplify onnx model
    conf_thres = 0.25 # CONF threshoud for NMS plugin
    iou_thres = 0.65 # IOU threshoud for NMS plugin

    ONNX_Export(weight_pt, ONNX_Path, device, input_shape, opset, topk, sim, conf_thres, iou_thres)


def ONNX_Export(weight_pt, weight_ONNX, device, input_shape, opset, topk, sim, conf_thres, iou_thres):
    PostDetect.conf_thres = conf_thres
    PostDetect.iou_thres = iou_thres
    PostDetect.topk = topk

    batch = input_shape[0]

    YOLOv8 = YOLO(weight_pt)

    model = YOLOv8.model.fuse().eval()
    for m in model.modules():
        optim(m)
        m.to(device)
    model.to(device)

    fake_input = torch.randn(input_shape).to(device)

    for _ in range(2):
        model(fake_input)

    with BytesIO() as f:
        torch.onnx.export(
            model, fake_input, f,
            opset_version=opset,
            input_names=['images'],
            output_names=['num_dets', 'bboxes', 'scores', 'labels'])
        f.seek(0)
        onnx_model = onnx.load(f)
    onnx.checker.check_model(onnx_model)

    shapes = [batch, 1, batch, topk, 4, batch, topk, batch, topk]

    for i in onnx_model.graph.output:
        for j in i.type.tensor_type.shape.dim:
            j.dim_param = str(shapes.pop(0))

    if sim:
        try:
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
        except Exception as e:
            print(f'Simplifier failure: {e}')

    onnx.save(onnx_model, weight_ONNX)
    print(f'-----> ONNX export success, saved as: ---> {weight_ONNX}')


class PostDetect(nn.Module):
    export = True
    shape = None
    dynamic = False
    iou_thres = 0.65
    conf_thres = 0.25
    topk = 100

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        shape = x[0].shape
        b, res, b_reg_num = shape[0], [], self.reg_max * 4
        for i in range(self.nl):
            res.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(
                0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        x = [i.view(b, self.no, -1) for i in res]
        y = torch.cat(x, 2)
        boxes, scores = y[:, :b_reg_num, ...], y[:, b_reg_num:, ...].sigmoid()
        boxes = boxes.view(b, 4, self.reg_max, -1).permute(0, 1, 3, 2)
        boxes = boxes.softmax(-1) @ torch.arange(self.reg_max).to(boxes)
        boxes0, boxes1 = -boxes[:, :2, ...], boxes[:, 2:, ...]
        boxes = self.anchors.repeat(b, 2, 1) + torch.cat([boxes0, boxes1], 1)
        boxes = boxes * self.strides

        return TRT_NMS.apply(boxes.transpose(1, 2), scores.transpose(1, 2), self.iou_thres, self.conf_thres, self.topk)

def make_anchors(feats: Tensor, strides: Tensor, grid_cell_offset: float = 0.5) -> Tuple[Tensor, Tensor]:
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


class TRT_NMS(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx: Graph,
            boxes: Tensor,
            scores: Tensor,
            iou_threshold: float = 0.65,
            score_threshold: float = 0.25,
            max_output_boxes: int = 100,
            background_class: int = -1,
            box_coding: int = 0,
            plugin_version: str = '1',
            score_activation: int = 0
            ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size, num_boxes, num_classes = scores.shape
        num_dets = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        boxes = torch.randn(batch_size, max_output_boxes, 4)
        scores = torch.randn(batch_size, max_output_boxes)
        labels = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)

        return num_dets, boxes, scores, labels

    @staticmethod
    def symbolic(
            g,
            boxes: Value,
            scores: Value,
            iou_threshold: float = 0.45,
            score_threshold: float = 0.25,
            max_output_boxes: int = 100,
            background_class: int = -1,
            box_coding: int = 0,
            score_activation: int = 0,
            plugin_version: str = '1'
            ) -> Tuple[Value, Value, Value, Value]:
        out = g.op('TRT::EfficientNMS_TRT',
                   boxes,
                   scores,
                   iou_threshold_f=iou_threshold,
                   score_threshold_f=score_threshold,
                   max_output_boxes_i=max_output_boxes,
                   background_class_i=background_class,
                   box_coding_i=box_coding,
                   plugin_version_s=plugin_version,
                   score_activation_i=score_activation,
                   outputs=4)
        nums_dets, boxes, scores, classes = out
        return nums_dets, boxes, scores, classes

def optim(module: nn.Module):
    s = str(type(module))[6:-2].split('.')[-1]
    if s == 'Detect':
        setattr(module, '__class__', PostDetect)
    elif s == 'C2f':
        setattr(module, '__class__', C2f)

class C2f(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        x = self.cv1(x)
        x = [x, x[:, self.c:, ...]]
        x.extend(m(x[-1]) for m in self.m)
        x.pop(1)
        return self.cv2(torch.cat(x, 1))
    
if __name__ == '__main__':
    main()
