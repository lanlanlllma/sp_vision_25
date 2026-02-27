import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from typing import Optional
from rknn.api import RKNN

ONNX_MODEL = './0526_rknn.onnx'
IMG_PATH = '/home/ma/rknn-toolkit2/rknn_transform/armor_dataset_reduced/1.png'
DATASET = './dataset.txt'

QUANTIZE_ON = True
# QUANTIZE_ON = False
# QUANTIZE_TYPE = 'w16a16i_dfp'  # 'w16a16i' or 'w16a16i_dfp' or 'w8a8'
# QUANTIZE_TYPE = 'w16a16i'  # 'w16a16i' or 'w16a16i_dfp' or 'w8a8'
QUANTIZE_TYPE = 'w8a8'  # 'w16a16i' or 'w16a16i_dfp' or 'w8a8'
RUN_BATCH_SIZE = 1
RKNN_MODEL = f'modles/b{RUN_BATCH_SIZE}_{QUANTIZE_ON}_{"none" if not QUANTIZE_ON else QUANTIZE_TYPE}_{ONNX_MODEL.split("/")[-1].replace(".onnx", ".rknn")}' 

OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = 640

# ---------------------------
# Single-head postprocess config
# ---------------------------
# For outputs shaped like [1, 25200, 22]. If your model layout differs, adjust indices.
CONF_THRESH = 0.65

OUTPUT_KEYPOINTS = 4
OUTPUT_KPT_START = 0
# Match OpenvinoInfer.cpp layout:
# 0~7: keypoints (4 points => 8 values)
# 8:   confidence (needs sigmoid)
# 9~12: color scores (4)
# 13~21: number/digit scores (9)
OUTPUT_CONF_INDEX = 8
OUTPUT_COLOR_START = 9
OUTPUT_COLOR_END = 13
OUTPUT_NUM_START = 13
OUTPUT_NUM_END = 22

# Color filter semantics (match OpenvinoInfer.cpp):
#   None -> keep all (but still drop color_id in {2,3})
#   0 -> detect blue only
#   1 -> detect red only
DETECT_COLOR = None


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def preprocess_bgr_uint8_hwc(img_bgr: np.ndarray, img_size: int) -> np.ndarray:
    """Input: BGR uint8 HWC -> Output: RGB float32 NCHW normalized (/255)."""
    if img_bgr is None:
        raise ValueError('img is None (cv2.imread failed)')
    if img_bgr.dtype != np.uint8:
        img_bgr = img_bgr.astype(np.uint8)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img = img_rgb.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, 0)  # NCHW
    return img


def _xyxy_iou(one_box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one box (4,) and many boxes (N,4), all xyxy."""
    xx1 = np.maximum(one_box[0], boxes[:, 0])
    yy1 = np.maximum(one_box[1], boxes[:, 1])
    xx2 = np.minimum(one_box[2], boxes[:, 2])
    yy2 = np.minimum(one_box[3], boxes[:, 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    area1 = np.maximum(0.0, one_box[2] - one_box[0]) * np.maximum(0.0, one_box[3] - one_box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-9
    return inter / union


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """NMS over xyxy boxes. Returns kept indices."""
    if boxes.size == 0:
        return np.array([], dtype=np.int64)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = _xyxy_iou(boxes[i], boxes[order[1:]])
        inds = np.where(ious < iou_thresh)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int64)


def postprocess_single_head(
    output: np.ndarray,
    conf_thresh: float = CONF_THRESH,
    nms_thresh: float = NMS_THRESH,
    detect_color: Optional[int] = DETECT_COLOR,
    kpt_start: int = OUTPUT_KPT_START,
    num_keypoints: int = OUTPUT_KEYPOINTS,
    conf_index: int = OUTPUT_CONF_INDEX,
    color_start: int = OUTPUT_COLOR_START,
    color_end: int = OUTPUT_COLOR_END,
    num_start: int = OUTPUT_NUM_START,
    num_end: int = OUTPUT_NUM_END,
):
    """Pipeline for output shaped like [1, 25200, 22].

    Steps:
      1) sigmoid(conf)
      2) conf threshold
      3) color/class argmax
      4) filter by detect_color
      5) parse keypoints + geometry
      6) bbox from keypoints
      7) NMS

    Returns tmp_objects: list[dict]
    """
    if output is None:
        return []
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError(f'Unexpected output shape: {output.shape}, expected [1, N, C]')

    preds = output[0].astype(np.float32)  # [N, C]
    c = preds.shape[1]
    if not (0 <= kpt_start < c):
        raise ValueError('kpt_start out of range')
    if not (0 <= conf_index < c):
        raise ValueError('conf_index out of range')
    if not (0 <= color_start < c and 0 < color_end <= c and color_start < color_end):
        raise ValueError('color_start/color_end out of range')
    if not (0 <= num_start < c and 0 < num_end <= c and num_start < num_end):
        raise ValueError('num_start/num_end out of range')

    # 1. Sigmoid confidence
    conf = sigmoid(preds[:, conf_index])

    # 2. Threshold
    keep = conf > conf_thresh
    if not np.any(keep):
        return []
    preds = preds[keep]
    conf = conf[keep]

    # 3. Decode color/num (argmax like C++ minMaxLoc)
    color_scores = preds[:, color_start:color_end]
    num_scores = preds[:, num_start:num_end]
    if color_scores.shape[1] != (color_end - color_start):
        raise ValueError('unexpected color_scores shape')
    if num_scores.shape[1] != (num_end - num_start):
        raise ValueError('unexpected num_scores shape')
    color_id = np.argmax(color_scores, axis=1).astype(np.int32)
    num_id = np.argmax(num_scores, axis=1).astype(np.int32)
    score_color = np.max(color_scores, axis=1).astype(np.float32)
    score_num = np.max(num_scores, axis=1).astype(np.float32)

    # 4. Color filter (match C++)
    # Always drop None/Purple => ids 2,3
    keep2 = (color_id != 2) & (color_id != 3)
    # detect_color: 0 blue-only, 1 red-only
    if detect_color is not None:
        keep2 = keep2 & (color_id == int(detect_color))
    if not np.any(keep2):
        return []
    preds = preds[keep2]
    conf = conf[keep2]
    color_id = color_id[keep2]
    num_id = num_id[keep2]
    score_color = score_color[keep2]
    score_num = score_num[keep2]

    # 5. Keypoints
    kpt_flat = preds[:, kpt_start:kpt_start + num_keypoints * 2]
    if kpt_flat.shape[1] != num_keypoints * 2:
        raise ValueError('Not enough channels for keypoints')
    kpts = kpt_flat.reshape(-1, num_keypoints, 2)

    # 6. BBox from keypoints
    x_min = np.min(kpts[:, :, 0], axis=1)
    y_min = np.min(kpts[:, :, 1], axis=1)
    x_max = np.max(kpts[:, :, 0], axis=1)
    y_max = np.max(kpts[:, :, 1], axis=1)
    boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)

    # 7. NMS
    # C++ uses NMSBoxes(boxes, confidences=score_num, score_threshold=conf_thresh)
    keep3 = score_num > conf_thresh
    if not np.any(keep3):
        return []
    boxes = boxes[keep3]
    conf = conf[keep3]
    color_id = color_id[keep3]
    num_id = num_id[keep3]
    kpts = kpts[keep3]
    score_color = score_color[keep3]
    score_num = score_num[keep3]

    keep_nms = nms_xyxy(boxes, score_num, nms_thresh)
    boxes = boxes[keep_nms]
    conf = conf[keep_nms]
    color_id = color_id[keep_nms]
    num_id = num_id[keep_nms]
    kpts = kpts[keep_nms]
    score_color = score_color[keep_nms]
    score_num = score_num[keep_nms]

    tmp_objects = []
    for box, prob, cid, nid, sc, sn, kp in zip(boxes, conf, color_id, num_id, score_color, score_num, kpts):
        p0, p1 = kp[0], kp[1]
        dx, dy = float(p1[0] - p0[0]), float(p1[1] - p0[1])
        angle = float(np.degrees(np.arctan2(dy, dx)))
        geom = {
            'center': (float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0)),
            'wh': (float(box[2] - box[0]), float(box[3] - box[1])),
            'angle_deg': angle,
        }
        tmp_objects.append({
            'bbox_xyxy': tuple(map(float, box)),
            'prob': float(prob),
            'score_num': float(sn),
            'score_color': float(sc),
            'color_id': int(cid),
            'label': int(nid),
            'keypoints': [tuple(map(float, pt)) for pt in kp],
            'geometry': geom,
        })
    return tmp_objects

CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")



def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def process(input, mask, anchors):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = input[..., 4]
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = input[..., 5:]

    box_xy = input[..., :2]*2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(IMG_SIZE/grid_h)

    box_wh = pow(input[..., 2:4]*2, 2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    boxes = boxes.reshape(-1, 4)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    _box_pos = np.where(box_confidences >= OBJ_THRESH)
    boxes = boxes[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score >= OBJ_THRESH)

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score* box_confidences)[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov5_post_process(input_data):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
               [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for input, mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, boxes, scores, classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    print("{:^12} {:^12}  {}".format('class', 'score', 'xmin, ymin, xmax, ymax'))
    print('-' * 50)
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

        print("{:^12} {:^12.3f} [{:>4}, {:>4}, {:>4}, {:>4}]".format(CLASSES[cl], score, top, left, right, bottom))

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    # RKNN quantized models typically expect uint8 input; use RKNN's built-in
    # normalization (divide by 255) to match the model's training pipeline.
    # rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588',model_pruning=True)
    # rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588',model_pruning=True,quantized_algorithm='mmse',optimization_level=2)
    # rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588',model_pruning=True,optimization_level=3,quantized_dtype="w16a16i_dfp")
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588',model_pruning=True,optimization_level=3,quantized_dtype=QUANTIZE_TYPE)
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET, auto_hybrid=True,rknn_batch_size=RUN_BATCH_SIZE)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    rknn.accuracy_analysis(inputs=[IMG_PATH])
    

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Set inputs
    img_bgr = cv2.imread(IMG_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f'Failed to read image: {IMG_PATH}')
    vis_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    # For RKNN inference: keep uint8 RGB NHWC.
    img_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    img_nhwc_u8 = np.expand_dims(img_rgb, 0)

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img_nhwc_u8], data_format=['nhwc'])
    # np.save('./onnx_yolov5_0.npy', outputs[0])
    if(RUN_BATCH_SIZE==1):
        # 3-branch output: [1,66,80,80], [1,66,40,40], [1,66,20,20]
        # Decode anchors and grid, then merge into [N, 22] for postprocess_single_head
        anchors = np.array([[10, 13], [16, 30], [33, 23],
                            [30, 61], [62, 45], [59, 119],
                            [116, 90], [156, 198], [373, 326]], dtype=np.float32)
        masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        strides = [8, 16, 32]
        num_anchors = 3
        num_channels = 22  # per anchor

        all_decoded = []
        for i, (out_raw, mask, stride) in enumerate(zip(outputs, masks, strides)):
            out = np.asarray(out_raw, dtype=np.float32)
            # out shape: [1, 66, H, W]  (66 = 3 anchors * 22 channels)
            _, c, h, w = out.shape
            assert c == num_anchors * num_channels, f'Expected {num_anchors*num_channels} channels, got {c}'

            # Reshape to [1, 3, 22, H, W] -> [1, 3, H, W, 22]
            out = out.reshape(1, num_anchors, num_channels, h, w)
            out = np.transpose(out, (0, 1, 3, 4, 2))  # [1, 3, H, W, 22]

            # Build grid
            grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            grid = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)  # [H, W, 2]
            grid = grid.reshape(1, 1, h, w, 2)  # broadcast over batch and anchors

            # Anchor sizes for this scale
            anchor_wh = np.array([anchors[m] for m in mask], dtype=np.float32)  # [3, 2]
            anchor_wh = anchor_wh.reshape(1, num_anchors, 1, 1, 2)

            # Decode formula (from original ONNX model constants):
            # The original detect head does: output = raw * mul_constant + add_constant
            # For keypoint channels (0-7): decoded = raw * anchor_wh + grid * stride
            #   This is a LINEAR decode, NOT the standard YOLOv5 sigmoid decode!
            # For conf/class channels (8-21): decoded = raw * 1.0 + 0.0 (pass-through)

            # Decode keypoints (channels 0-7): 4 points, each (x, y)
            kpts_raw = out[..., 0:8]  # [1, 3, H, W, 8]
            kpts_decoded = np.zeros_like(kpts_raw)
            for kp_idx in range(4):
                kx = kpts_raw[..., kp_idx*2]
                ky = kpts_raw[..., kp_idx*2+1]
                # Linear decode: kpt = raw * anchor_scale + grid_offset
                kpts_decoded[..., kp_idx*2] = kx * anchor_wh[..., 0] + grid[..., 0] * stride
                kpts_decoded[..., kp_idx*2+1] = ky * anchor_wh[..., 1] + grid[..., 1] * stride
            # Channels 8-21 stay raw (sigmoid applied later in postprocess)
            rest = out[..., 8:]  # [1, 3, H, W, 14]

            # Combine decoded keypoints + raw rest
            decoded = np.concatenate([kpts_decoded, rest], axis=-1)  # [1, 3, H, W, 22]

            # Flatten to [N, 22]
            decoded = decoded.reshape(-1, num_channels)  # [3*H*W, 22]
            all_decoded.append(decoded)

        # Merge all scales
        merged = np.concatenate(all_decoded, axis=0)  # [25200, 22]
        merged = np.expand_dims(merged, 0)  # [1, 25200, 22]
        print(f'Decoded merged output shape: {merged.shape}')

        # Save raw decoded output
        out2d = merged[0].astype(np.float32)
        np.savetxt('./rknn_yolov5_0.txt', out2d, fmt='%.6g', delimiter=' ')
        print('Saved decoded output to rknn_yolov5_0.txt')

        # Post process (reuse single-head postprocess)
        tmp_objects = postprocess_single_head(
            merged,
            conf_thresh=CONF_THRESH,
            nms_thresh=NMS_THRESH,
            detect_color=DETECT_COLOR,
        )
        print(f'Postprocess done, got {len(tmp_objects)} objects')
        for i, obj in enumerate(tmp_objects[:20]):
            print(i, obj)

        # Visualization
        for obj in tmp_objects:
            x1, y1, x2, y2 = map(int, obj['bbox_xyxy'])
            cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for (x, y) in obj['keypoints']:
                cv2.circle(vis_bgr, (int(x), int(y)), 2, (0, 0, 255), -1)
            prob = float(obj.get('prob', obj.get('score', 0.0)))
            label = int(obj.get('label', -1))
            cv2.putText(
                vis_bgr,
                f"cid={obj['color_id']} cls={label} p={prob:.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
        cv2.imwrite('result.jpg', vis_bgr)
        print('Save results to result.jpg!')
    else:
        print(f'Batch size {RUN_BATCH_SIZE} inference done.')

    rknn.release()
