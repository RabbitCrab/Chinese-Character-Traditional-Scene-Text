import numpy as np
import torch
import cv2
import torch.backends.cudnn as cudnn
import argparse
import sys
from pathlib import Path
import csv
import os

# yolo import
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import check_img_size, check_imshow, check_requirements, check_suffix, colorstr, is_ascii, \
    non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, \
    save_one_box
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync

# resnet import
# from resnet_training.train_resnet import TextModel
from resnet_training.train_efficientnet import TextModel
import json

from functools import reduce
import operator
import math


num_to_word = json.load(open('resnet_training/new_num_to_word.json', 'r', encoding='utf-8'))

csv_content = []
with open('data/public/Task2_Public_String_Coordinate.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        processed_row = []
        for i in row:
            processed_row.append(i)
        csv_content.append(processed_row)

final_output = []

def compareArea(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5):
    l12 = ((((x2 - x1) ** 2) + ((y2 - y1) ** 2)) ** 0.5)
    l23 = ((((x2 - x3) ** 2) + ((y2 - y3) ** 2)) ** 0.5)
    l34 = ((((x4 - x3) ** 2) + ((y4 - y3) ** 2)) ** 0.5)
    l14 = ((((x4 - x1) ** 2) + ((y4 - y1) ** 2)) ** 0.5)
    l13 = ((((x1 - x3) ** 2) + ((y1 - y3) ** 2)) ** 0.5)
    
    sHalf1 = (l12 + l23 + l13)/2
    areaHalf1 = np.sqrt(sHalf1 * (sHalf1 - l12) * (sHalf1 - l23) * (sHalf1 - l13))
    sHalf2 = (l14 + l34 + l13)/2
    areaHalf2 = np.sqrt(sHalf2 * (sHalf2 - l14) * (sHalf2 - l34) * (sHalf2 - l13))
    areaAll1 = areaHalf1 + areaHalf2

    l15 = ((((x5 - x1) ** 2) + ((y5 - y1) ** 2)) ** 0.5)
    l25 = ((((x5 - x2) ** 2) + ((y5 - y2) ** 2)) ** 0.5)
    l35 = ((((x5 - x3) ** 2) + ((y5 - y3) ** 2)) ** 0.5)
    l45 = ((((x5 - x4) ** 2) + ((y5 - y4) ** 2)) ** 0.5)

    sQuarter125 = (l12 + l15 + l25)/2
    areaQuarter125 = np.sqrt(sQuarter125 * (sQuarter125 - l12) * (sQuarter125 - l15) * (sQuarter125 - l25))
    sQuarter235 = (l23 + l35 + l25)/2
    areaQuarter235 = np.sqrt(sQuarter235 * (sQuarter235 - l23) * (sQuarter235 - l35) * (sQuarter235 - l25))
    sQuarter345 = (l34 + l35 + l45)/2
    areaQuarter345 = np.sqrt(sQuarter345 * (sQuarter345 - l34) * (sQuarter345 - l35) * (sQuarter345 - l45))
    sQuarter145 = (l14 + l15 + l45)/2
    areaQuarter145 = np.sqrt(sQuarter145 * (sQuarter145 - l14) * (sQuarter145 - l15) * (sQuarter145 - l45))
    areaAll2 = areaQuarter125 + areaQuarter235 + areaQuarter345 + areaQuarter145

    return abs(areaAll2 - areaAll1) < 1e-6

# yolov5
@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)
    else:  # TensorFlow models
        check_requirements(('tensorflow>=2.4.1',))
        import tensorflow as tf
        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))

            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, 'rb').read())
            frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
        elif saved_model:
            model = tf.keras.models.load_model(w)
        elif tflite:
            interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)

    # Text recognition model
    text_model = TextModel(5038)  # text model of small image size
    text_model.load_state_dict(torch.load('resnet_training/eff_final_no_val_imgsz64.pt'))
    text_model.eval()

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0
    count = 0
    for path, img, im0s, vid_cap in dataset:
        t1 = time_sync()
        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        if pt:
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
        else:  # tensorflow model (tflite, pb, saved_model)
            imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
            if pb:
                pred = frozen_func(x=tf.constant(imn)).numpy()
            elif saved_model:
                pred = model(imn, training=False).numpy()
            elif tflite:
                if int8:
                    scale, zero_point = input_details[0]['quantization']
                    imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                interpreter.set_tensor(input_details[0]['index'], imn)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                if int8:
                    scale, zero_point = output_details[0]['quantization']
                    pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
            pred[..., 0] *= imgsz[1]  # x
            pred[..., 1] *= imgsz[0]  # y
            pred[..., 2] *= imgsz[1]  # w
            pred[..., 3] *= imgsz[0]  # h
            pred = torch.tensor(pred)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                while path.split(os.sep)[-1].split('.')[0] == csv_content[count][0]:
                    # the targeted region
                    _, x1, y1, x2, y2, x3, y3, x4, y4 = csv_content[count]
                    x1 = int(x1)
                    x2 = int(x2)
                    x3 = int(x3)
                    x4 = int(x4)
                    y1 = int(y1)
                    y2 = int(y2)
                    y3 = int(y3)
                    y4 = int(y4)
                    coords = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
                    coords = sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
                    x1, y1 = coords[0]
                    x2, y2 = coords[1]
                    x3, y3 = coords[2]
                    x4, y4 = coords[3]
                    # Extract text image and feed to recognition model
                    texts = []
                    cor = det[:, :4]
                    sorted_cor = sorted(cor, key=lambda s: s[0] + s[1])
                    targeted_texts = []
                    for xyxy in sorted_cor:
                        x1_, y1_, x2_, y2_ = xyxy
                        x1_ = int(x1_.item())
                        x2_ = int(x2_.item())
                        y1_ = int(y1_.item())
                        y2_ = int(y2_.item())
                        center_x = (x1_ + x2_) / 2
                        center_y = (y1_ + y2_) / 2
                        # predicted box inside the targeted region
                        if compareArea(x1, y1, x2, y2, x3, y3, x4, y4, center_x, center_y):
                            targeted_texts.append(xyxy)
                    for xyxy in targeted_texts:
                        x1, y1, x2, y2 = xyxy
                        x1 = int(x1.item())
                        x2 = int(x2.item())
                        y1 = int(y1.item())
                        y2 = int(y2.item())
                        text_img = im0[y1:y2, x1:x2]
                        text_img = cv2.resize(text_img, (64, 64))
                        text_img = torch.from_numpy(text_img.transpose(2, 0, 1)).float()
                        text_img = text_img[None, :, :, :]
                        pred = text_model(text_img)
                        texts.append(num_to_word[str(pred.argmax(dim=1).item())])
                    if len(texts):
                        result = ''
                        for t in texts:
                            result += t
                    else:  # no word contained in this region
                        result = '###'
                    temp = csv_content[count]
                    temp.append(result)
                    print(temp)
                    final_output.append(temp)
                    count += 1
                    if count >= len(csv_content):
                        break
                    # print predicted result
                    # print(result)
            else:
                # this image doesn't contain any word
                while path.split(os.sep)[-1].split('.')[0] == csv_content[count][0]:
                    result = '###'
                    temp = csv_content[count]
                    temp.append(result)
                    print(temp)
                    final_output.append(temp)
                    count += 1
                    if count >= len(csv_content):
                        break
                    # print predicted result
                    # print(result)
    with open('output_final.csv', 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.writer(output_file)
        writer.writerows(final_output)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[160], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
