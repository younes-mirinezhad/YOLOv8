from ultralytics import YOLO
import torch
import torch.nn as nn
from torch import Graph, Tensor, Value
from typing import Tuple, Union, Optional, List, OrderedDict
from io import BytesIO
import onnx
try: 
    import onnxsim 
except ImportError: 
    onnxsim = None
from pathlib import Path
import tensorrt as trt
import pickle
import numpy as np
import os

def main():
    weight_pt = "/home/chiko/Downloads/Compressed/yolov8n.pt" # PyTorch yolov8 weights
    weight_ONNX = weight_pt.replace(".pt", "_end2end.onnx") # ONNX yolov8 weights
    device = "cuda:0" # cpu , cuda:0 --> Export device
    input_shape = [1, 3, 960, 960] # Model input shape only for api builder
    topk = 100 # Max number of detection bboxes
    conf_thres = 0.25 # CONF threshoud for NMS plugin
    iou_thres = 0.65 # IOU threshoud for NMS plugin

    opset = 11 # ONNX opset version
    sim = True # simplify onnx model
    ONNX_Export(weight_pt, weight_ONNX, device, input_shape, opset, topk, sim, conf_thres, iou_thres)

    seg = False # Build seg model by onnx
    fp16 = True # Build model with fp16 mode
    Engine_Export(weight_ONNX, device, seg, fp16, input_shape, iou_thres, conf_thres, topk)
    
    os.remove(weight_ONNX)

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
    print(f'ONNX export success, saved as {weight_ONNX}')

def Engine_Export(weight_ONNX, device, seg, fp16, input_shape, iou_thres, conf_thres, topk):
    builder = EngineBuilder(weight_ONNX, device)
    builder.seg = seg
    builder.build(fp16=fp16, input_shape=input_shape, iou_thres=iou_thres, conf_thres=conf_thres, topk=topk)

def optim(module: nn.Module):
    s = str(type(module))[6:-2].split('.')[-1]
    if s == 'Detect':
        setattr(module, '__class__', PostDetect)
    elif s == 'C2f':
        setattr(module, '__class__', C2f)

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

class C2f(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        x = self.cv1(x)
        x = [x, x[:, self.c:, ...]]
        x.extend(m(x[-1]) for m in self.m)
        x.pop(1)
        return self.cv2(torch.cat(x, 1))

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

class EngineBuilder:
    seg = False

    def __init__(self, checkpoint: Union[str, Path], device: Optional[Union[str, int, torch.device]] = None) -> None:
        checkpoint = Path(checkpoint) if isinstance(checkpoint, str) else checkpoint
        assert checkpoint.exists() and checkpoint.suffix in ('.onnx', '.pkl')
        self.api = checkpoint.suffix == '.pkl'
        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device(f'cuda:{device}')

        self.checkpoint = checkpoint
        self.device = device

    def __build_engine(self,
                       fp16: bool = True,
                       input_shape: Union[List, Tuple] = (1, 3, 640, 640),
                       iou_thres: float = 0.65,
                       conf_thres: float = 0.25,
                       topk: int = 100,
                       with_profiling: bool = True) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace='')
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = torch.cuda.get_device_properties(
            self.device).total_memory
        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)

        self.logger = logger
        self.builder = builder
        self.network = network
        if self.api:
            self.build_from_api(fp16, input_shape, iou_thres, conf_thres, topk)
        else:
            self.build_from_onnx(iou_thres, conf_thres, topk)
        if fp16 and self.builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        self.weight = self.checkpoint.with_suffix('.engine')

        if with_profiling:
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        with self.builder.build_engine(self.network, config) as engine:
            self.weight.write_bytes(engine.serialize())
        self.logger.log(trt.Logger.WARNING, f'Build tensorrt engine finish.\n' f'Save in {str(self.weight.absolute())}')

    def build(self,
              fp16: bool = True,
              input_shape: Union[List, Tuple] = (1, 3, 640, 640),
              iou_thres: float = 0.65,
              conf_thres: float = 0.25,
              topk: int = 100,
              with_profiling=True) -> None:
        self.__build_engine(fp16, input_shape, iou_thres, conf_thres, topk, with_profiling)

    def build_from_onnx(self, iou_thres: float = 0.65, conf_thres: float = 0.25, topk: int = 100):
        parser = trt.OnnxParser(self.network, self.logger)
        onnx_model = onnx.load(str(self.checkpoint))
        if not self.seg:
            onnx_model.graph.node[-1].attribute[2].i = topk
            onnx_model.graph.node[-1].attribute[3].f = conf_thres
            onnx_model.graph.node[-1].attribute[4].f = iou_thres

        if not parser.parse(onnx_model.SerializeToString()):
            raise RuntimeError(f'failed to load ONNX file: {str(self.checkpoint)}')
        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        for inp in inputs:
            self.logger.log(trt.Logger.WARNING,
                f'input "{inp.name}" with shape: {inp.shape} '
                f'dtype: {inp.dtype}')
        for out in outputs:
            self.logger.log(trt.Logger.WARNING,
                f'output "{out.name}" with shape: {out.shape} '
                f'dtype: {out.dtype}')

    def build_from_api(self, fp16: bool = True, input_shape: Union[List, Tuple] = (1, 3, 640, 640),
        iou_thres: float = 0.65, conf_thres: float = 0.25, topk: int = 100,
    ):
        assert not self.seg

        with open(self.checkpoint, 'rb') as f:
            state_dict = pickle.load(f)
        mapping = {0.25: 1024, 0.5: 1024, 0.75: 768, 1.0: 512, 1.25: 512}

        GW = state_dict['GW']
        GD = state_dict['GD']
        width_64 = self.get_width(64, GW)
        width_128 = self.get_width(128, GW)
        width_256 = self.get_width(256, GW)
        width_512 = self.get_width(512, GW)
        width_1024 = self.get_width(mapping[GW], GW)
        depth_3 = self.get_depth(3, GD)
        depth_6 = self.get_depth(6, GD)
        strides = state_dict['strides']
        reg_max = state_dict['reg_max']
        images = self.network.add_input(name='images', dtype=trt.float32, shape=trt.Dims4(input_shape))
        assert images, 'Add input failed'

        Conv_0 = self.Conv(self.network, state_dict, images, width_64, 3, 2, 1, 'Conv.0')
        Conv_1 = self.Conv(self.network, state_dict, Conv_0.get_output(0), width_128, 3, 2, 1, 'Conv.1')
        C2f_2 = self.C2f(self.network, state_dict, Conv_1.get_output(0), width_128, depth_3, True, 1, 0.5, 'C2f.2')
        Conv_3 = self.Conv(self.network, state_dict, C2f_2.get_output(0), width_256, 3, 2, 1, 'Conv.3')
        C2f_4 = self.C2f(self.network, state_dict, Conv_3.get_output(0), width_256, depth_6, True, 1, 0.5, 'C2f.4')
        Conv_5 = self.Conv(self.network, state_dict, C2f_4.get_output(0), width_512, 3, 2, 1, 'Conv.5')
        C2f_6 = self.C2f(self.network, state_dict, Conv_5.get_output(0), width_512, depth_6, True, 1, 0.5, 'C2f.6')
        Conv_7 = self.Conv(self.network, state_dict, C2f_6.get_output(0), width_1024, 3, 2, 1, 'Conv.7')
        C2f_8 = self.C2f(self.network, state_dict, Conv_7.get_output(0), width_1024, depth_3, True, 1, 0.5, 'C2f.8')
        SPPF_9 = self.SPPF(self.network, state_dict, C2f_8.get_output(0), width_1024, width_1024, 5, 'SPPF.9')
        Upsample_10 = self.network.add_resize(SPPF_9.get_output(0))
        assert Upsample_10, 'Add Upsample_10 failed'
        Upsample_10.resize_mode = trt.ResizeMode.NEAREST
        Upsample_10.shape = Upsample_10.get_output(0).shape[:2] + C2f_6.get_output(0).shape[2:]
        input_tensors11 = [Upsample_10.get_output(0), C2f_6.get_output(0)]
        Cat_11 = self.network.add_concatenation(input_tensors11)
        C2f_12 = C2f(self.network, state_dict, Cat_11.get_output(0), width_512, depth_3, False, 1, 0.5, 'C2f.12')
        Upsample13 = self.network.add_resize(C2f_12.get_output(0))
        assert Upsample13, 'Add Upsample13 failed'
        Upsample13.resize_mode = trt.ResizeMode.NEAREST
        Upsample13.shape = Upsample13.get_output(0).shape[:2] + C2f_4.get_output(0).shape[2:]
        input_tensors14 = [Upsample13.get_output(0), C2f_4.get_output(0)]
        Cat_14 = self.network.add_concatenation(input_tensors14)
        C2f_15 = C2f(self.network, state_dict, Cat_14.get_output(0), width_256,depth_3, False, 1, 0.5, 'C2f.15')
        Conv_16 = self.Conv(self.network, state_dict, C2f_15.get_output(0),width_256, 3, 2, 1, 'Conv.16')
        input_tensors17 = [Conv_16.get_output(0), C2f_12.get_output(0)]
        Cat_17 = self.network.add_concatenation(input_tensors17)
        C2f_18 = C2f(self.network, state_dict, Cat_17.get_output(0), width_512, depth_3, False, 1, 0.5, 'C2f.18')
        Conv_19 = self.Conv(self.network, state_dict, C2f_18.get_output(0), width_512, 3, 2, 1, 'Conv.19')
        input_tensors20 = [Conv_19.get_output(0), SPPF_9.get_output(0)]
        Cat_20 = self.network.add_concatenation(input_tensors20)
        C2f_21 = C2f(self.network, state_dict, Cat_20.get_output(0), width_1024, depth_3, False, 1, 0.5, 'C2f.21')
        input_tensors22 = [C2f_15.get_output(0), C2f_18.get_output(0), C2f_21.get_output(0)]
        batched_nms = self.Detect(self.network, state_dict, input_tensors22, strides, 'Detect.22', reg_max, fp16, iou_thres, conf_thres, topk)
        for o in range(batched_nms.num_outputs):
            self.network.mark_output(batched_nms.get_output(o))

    def SPPF(self, network: trt.INetworkDefinition, weights: OrderedDict,
            input: trt.ITensor, c1: int, c2: int, ksize: int,
            layer_name: str) -> trt.ILayer:
        c_ = c1 // 2
        conv1 = self.Conv(network, weights, input, c_, 1, 1, 1, layer_name + '.cv1')

        pool1 = network.add_pooling_nd(conv1.get_output(0), trt.PoolingType.MAX, trt.DimsHW(ksize, ksize))
        assert pool1, 'Add pooling_nd layer failed'
        pool1.padding_nd = trt.DimsHW(ksize // 2, ksize // 2)
        pool1.stride_nd = trt.DimsHW(1, 1)

        pool2 = network.add_pooling_nd(pool1.get_output(0), trt.PoolingType.MAX, trt.DimsHW(ksize, ksize))
        assert pool2, 'Add pooling_nd layer failed'
        pool2.padding_nd = trt.DimsHW(ksize // 2, ksize // 2)
        pool2.stride_nd = trt.DimsHW(1, 1)

        pool3 = network.add_pooling_nd(pool2.get_output(0), trt.PoolingType.MAX, trt.DimsHW(ksize, ksize))
        assert pool3, 'Add pooling_nd layer failed'
        pool3.padding_nd = trt.DimsHW(ksize // 2, ksize // 2)
        pool3.stride_nd = trt.DimsHW(1, 1)

        input_tensors = [conv1.get_output(0), pool1.get_output(0), pool2.get_output(0), pool3.get_output(0)]
        cat = network.add_concatenation(input_tensors)
        assert cat, 'Add concatenation layer failed'
        conv2 = self.Conv(network, weights, cat.get_output(0), c2, 1, 1, 1, layer_name + '.cv2')
        return conv2
    def C2f(self, network: trt.INetworkDefinition, weights: OrderedDict,
            input: trt.ITensor, cout: int, n: int, shortcut: bool, group: int,
            scale: float, layer_name: str) -> trt.ILayer:
        c_ = int(cout * scale)  # e:expand param
        conv1 = self.Conv(network, weights, input, 2 * c_, 1, 1, 1, layer_name + '.cv1')
        y1 = conv1.get_output(0)

        b, _, h, w = y1.shape
        slice = network.add_slice(y1, (0, c_, 0, 0), (b, c_, h, w), (1, 1, 1, 1))
        assert slice, 'Add slice layer failed'
        y2 = slice.get_output(0)

        input_tensors = [y1]
        for i in range(n):
            b = self.Bottleneck(network, weights, y2, c_, c_, shortcut, group, 1.0, layer_name + '.m.' + str(i))
            y2 = b.get_output(0)
            input_tensors.append(y2)

        cat = network.add_concatenation(input_tensors)
        assert cat, 'Add concatenation layer failed'

        conv2 = self.Conv(network, weights, cat.get_output(0), cout, 1, 1, 1, layer_name + '.cv2')
        return conv2
    def Conv(self, network: trt.INetworkDefinition, weights: OrderedDict,
            input: trt.ITensor, out_channel: int, ksize: int, stride: int,
            group: int, layer_name: str) -> trt.ILayer:
        padding = ksize // 2
        if ksize > 3:
            padding -= 1
        conv_w = self.trtweight(weights[layer_name + '.conv.weight'])
        conv_b = self.trtweight(weights[layer_name + '.conv.bias'])

        conv = network.add_convolution_nd(input,
                                        num_output_maps=out_channel,
                                        kernel_shape=trt.DimsHW(ksize, ksize),
                                        kernel=conv_w,
                                        bias=conv_b)
        assert conv, 'Add convolution_nd layer failed'
        conv.stride_nd = trt.DimsHW(stride, stride)
        conv.padding_nd = trt.DimsHW(padding, padding)
        conv.num_groups = group

        sigmoid = network.add_activation(conv.get_output(0), trt.ActivationType.SIGMOID)
        assert sigmoid, 'Add activation layer failed'
        dot_product = network.add_elementwise(conv.get_output(0), sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
        assert dot_product, 'Add elementwise layer failed'
        return dot_product
    def Conv2d(self, network: trt.INetworkDefinition, weights: OrderedDict,
               input: trt.ITensor, out_channel: int, ksize: int, stride: int,
               group: int, layer_name: str) -> trt.ILayer:
        padding = ksize // 2
        conv_w = self.trtweight(weights[layer_name + '.weight'])
        conv_b = self.trtweight(weights[layer_name + '.bias'])
        conv = network.add_convolution_nd(input,
                                        num_output_maps=out_channel,
                                        kernel_shape=trt.DimsHW(ksize, ksize),
                                        kernel=conv_w,
                                        bias=conv_b)
        assert conv, 'Add convolution_nd layer failed'
        conv.stride_nd = trt.DimsHW(stride, stride)
        conv.padding_nd = trt.DimsHW(padding, padding)
        conv.num_groups = group
        return conv
    def Detect(self, 
        network: trt.INetworkDefinition,
        weights: OrderedDict,
        input: Union[List, Tuple],
        s: Union[List, Tuple],
        layer_name: str,
        reg_max: int = 16,
        fp16: bool = True,
        iou: float = 0.65,
        conf: float = 0.25,
        topk: int = 100,
    ) -> trt.ILayer:
        bboxes_branch = []
        scores_branch = []
        anchors = []
        strides = []
        for i, (inp, stride) in enumerate(zip(input, s)):
            h, w = inp.shape[2:]
            sx = np.arange(0, w).astype(np.float16 if fp16 else np.float32) + 0.5
            sy = np.arange(0, h).astype(np.float16 if fp16 else np.float32) + 0.5
            sy, sx = np.meshgrid(sy, sx)
            a = np.ascontiguousarray(np.stack((sy, sx), -1).reshape(-1, 2))
            anchors.append(a)
            strides.append(np.full((1, h * w), stride, dtype=np.float16 if fp16 else np.float32))
            c2 = weights[f'{layer_name}.cv2.{i}.0.conv.weight'].shape[0]
            c3 = weights[f'{layer_name}.cv3.{i}.0.conv.weight'].shape[0]
            nc = weights[f'{layer_name}.cv3.0.2.weight'].shape[0]
            reg_max_x4 = weights[layer_name + f'.cv2.{i}.2.weight'].shape[0]
            assert reg_max_x4 == reg_max * 4
            b_Conv_0 = self.Conv(network, weights, inp, c2, 3, 1, 1, layer_name + f'.cv2.{i}.0')
            b_Conv_1 = self.Conv(network, weights, b_Conv_0.get_output(0), c2, 3, 1, 1, layer_name + f'.cv2.{i}.1')
            b_Conv_2 = self.Conv2d(network, weights, b_Conv_1.get_output(0), reg_max_x4, 1, 1, 1, layer_name + f'.cv2.{i}.2')

            b_out = b_Conv_2.get_output(0)
            b_shape = network.add_constant([4,], np.array(b_out.shape[0:1] + (4, reg_max, -1), dtype=np.int32))
            assert b_shape, 'Add constant layer failed'
            b_shuffle = network.add_shuffle(b_out)
            assert b_shuffle, 'Add shuffle layer failed'
            b_shuffle.set_input(1, b_shape.get_output(0))
            b_shuffle.second_transpose = (0, 3, 1, 2)

            bboxes_branch.append(b_shuffle.get_output(0))

            s_Conv_0 = self.Conv(network, weights, inp, c3, 3, 1, 1, layer_name + f'.cv3.{i}.0')
            s_Conv_1 = self.Conv(network, weights, s_Conv_0.get_output(0), c3, 3, 1, 1, layer_name + f'.cv3.{i}.1')
            s_Conv_2 = self.Conv2d(network, weights, s_Conv_1.get_output(0), nc, 1, 1, 1, layer_name + f'.cv3.{i}.2')
            s_out = s_Conv_2.get_output(0)
            s_shape = network.add_constant([3,], np.array(s_out.shape[0:2] + (-1, ), dtype=np.int32))
            assert s_shape, 'Add constant layer failed'
            s_shuffle = network.add_shuffle(s_out)
            assert s_shuffle, 'Add shuffle layer failed'
            s_shuffle.set_input(1, s_shape.get_output(0))
            s_shuffle.second_transpose = (0, 2, 1)

            scores_branch.append(s_shuffle.get_output(0))

        Cat_bboxes = network.add_concatenation(bboxes_branch)
        assert Cat_bboxes, 'Add concatenation layer failed'
        Cat_scores = network.add_concatenation(scores_branch)
        assert Cat_scores, 'Add concatenation layer failed'
        Cat_scores.axis = 1

        Softmax = network.add_softmax(Cat_bboxes.get_output(0))
        assert Softmax, 'Add softmax layer failed'
        Softmax.axes = 1 << 3

        SCORES = network.add_activation(Cat_scores.get_output(0), trt.ActivationType.SIGMOID)
        assert SCORES, 'Add activation layer failed'

        reg_max = np.arange(0, reg_max).astype(np.float16 if fp16 else np.float32).reshape((1, 1, -1, 1))
        constant = network.add_constant(reg_max.shape, reg_max)
        assert constant, 'Add constant layer failed'
        Matmul = network.add_matrix_multiply(Softmax.get_output(0), trt.MatrixOperation.NONE, constant.get_output(0), trt.MatrixOperation.NONE)
        assert Matmul, 'Add matrix_multiply layer failed'
        pre_bboxes = network.add_gather(Matmul.get_output(0), network.add_constant([1,], np.array([0], dtype=np.int32)).get_output(0), 3)
        assert pre_bboxes, 'Add gather layer failed'
        pre_bboxes.num_elementwise_dims = 1

        pre_bboxes_tensor = pre_bboxes.get_output(0)
        b, c, _ = pre_bboxes_tensor.shape
        slice_x1y1 = network.add_slice(pre_bboxes_tensor, (0, 0, 0), (b, c, 2), (1, 1, 1))
        assert slice_x1y1, 'Add slice layer failed'
        slice_x2y2 = network.add_slice(pre_bboxes_tensor, (0, 0, 2), (b, c, 2), (1, 1, 1))
        assert slice_x2y2, 'Add slice layer failed'
        anchors = np.concatenate(anchors, 0)[np.newaxis]
        anchors = network.add_constant(anchors.shape, anchors)
        assert anchors, 'Add constant layer failed'
        strides = np.concatenate(strides, 1)[..., np.newaxis]
        strides = network.add_constant(strides.shape, strides)
        assert strides, 'Add constant layer failed'

        Sub = network.add_elementwise(anchors.get_output(0), slice_x1y1.get_output(0), trt.ElementWiseOperation.SUB)
        assert Sub, 'Add elementwise layer failed'
        Add = network.add_elementwise(anchors.get_output(0), slice_x2y2.get_output(0), trt.ElementWiseOperation.SUM)
        assert Add, 'Add elementwise layer failed'
        x1y1 = Sub.get_output(0)
        x2y2 = Add.get_output(0)

        Cat_bboxes_ = network.add_concatenation([x1y1, x2y2])
        assert Cat_bboxes_, 'Add concatenation layer failed'
        Cat_bboxes_.axis = 2

        BBOXES = network.add_elementwise(Cat_bboxes_.get_output(0), strides.get_output(0), trt.ElementWiseOperation.PROD)
        assert BBOXES, 'Add elementwise layer failed'
        plugin_creator = trt.get_plugin_registry().get_plugin_creator('EfficientNMS_TRT', '1')
        assert plugin_creator, 'Plugin EfficientNMS_TRT is not registried'

        background_class = trt.PluginField('background_class', np.array(-1, np.int32), trt.PluginFieldType.INT32)
        box_coding = trt.PluginField('box_coding', np.array(0, np.int32), trt.PluginFieldType.INT32)
        iou_threshold = trt.PluginField('iou_threshold', np.array(iou, dtype=np.float32), trt.PluginFieldType.FLOAT32)
        max_output_boxes = trt.PluginField('max_output_boxes', np.array(topk, np.int32), trt.PluginFieldType.INT32)
        plugin_version = trt.PluginField('plugin_version', np.array('1'), trt.PluginFieldType.CHAR)
        score_activation = trt.PluginField('score_activation', np.array(0, np.int32), trt.PluginFieldType.INT32)
        score_threshold = trt.PluginField('score_threshold', np.array(conf, dtype=np.float32), trt.PluginFieldType.FLOAT32)

        batched_nms_op = plugin_creator.create_plugin( name='batched_nms',
            field_collection=trt.PluginFieldCollection([background_class, box_coding, iou_threshold, max_output_boxes, 
                                                        plugin_version, score_activation, score_threshold]))

        batched_nms = network.add_plugin_v2(inputs=[BBOXES.get_output(0), SCORES.get_output(0)], plugin=batched_nms_op)

        batched_nms.get_output(0).name = 'num_dets'
        batched_nms.get_output(1).name = 'bboxes'
        batched_nms.get_output(2).name = 'scores'
        batched_nms.get_output(3).name = 'labels'

        return batched_nms
    def get_width(self, x: int, gw: float, divisor: int = 8) -> int:
        return int(np.ceil(x * gw / divisor) * divisor)
    def get_depth(self, x: int, gd: float) -> int:
        return max(int(round(x * gd)), 1)
    def trtweight(self, weights: np.ndarray) -> trt.Weights:
        weights = weights.astype(weights.dtype.name)
        return trt.Weights(weights)
    def Bottleneck(self, network: trt.INetworkDefinition, weights: OrderedDict,
                input: trt.ITensor, c1: int, c2: int, shortcut: bool,
                group: int, scale: float, layer_name: str) -> trt.ILayer:
        c_ = int(c2 * scale)
        conv1 = self.Conv(network, weights, input, c_, 3, 1, 1, layer_name + '.cv1')
        conv2 = self.Conv(network, weights, conv1.get_output(0), c2, 3, 1, group, layer_name + '.cv2')
        if shortcut and c1 == c2:
            ew = network.add_elementwise(input, conv2.get_output(0), op=trt.ElementWiseOperation.SUM)
            assert ew, 'Add elementwise layer failed'
            return ew
        return conv2

if __name__ == '__main__':
    main()
