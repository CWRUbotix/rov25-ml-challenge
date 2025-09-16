from pathlib import Path
import numpy as np
from mean_average_precision import MetricBuilder

IMG_SIZE = (640, 640)  # adjust if your dataset uses another size

def yolo_to_xyxy(box, img_w, img_h):
    '''Convert YOLO normalized [xc, yc, w, h] to [x1, y1, x2, y2] absolute.'''
    x_c, y_c, w, h = box
    x1 = (x_c - w/2) * img_w
    y1 = (y_c - h/2) * img_h
    x2 = (x_c + w/2) * img_w
    y2 = (y_c + h/2) * img_h
    return [x1, y1, x2, y2]

def load_gt_boxes(filepath: Path, img_size):
    boxes = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = map(float, parts)
            xyxy = yolo_to_xyxy([x, y, w, h], *img_size)
            boxes.append(xyxy + [int(cls), 0, 0])  # difficult=0, crowd=0
    return np.array(boxes)

def load_pred_boxes(filepath: Path, img_size):
    boxes = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            cls, conf, x, y, w, h = map(float, parts)
            xyxy = yolo_to_xyxy([x, y, w, h], *img_size)
            boxes.append(xyxy + [int(cls), conf])
    return np.array(boxes)

GT_PATH = Path('dataset') / 'labels' / 'test'
PRED_PATH = Path('output') / 'effdet_predictions' / 'tf_efficientdet_d2_ap'

if __name__ == '__main__':
    maps50: list[float] = []
    maps5095: list[float] = []

    preds = list(PRED_PATH.glob('*.txt'))

    print(preds)

    for gt_filepath in GT_PATH.glob('*.txt'):
        pred_filepath = PRED_PATH / gt_filepath.name

        print(pred_filepath)

        gt_boxes = load_gt_boxes(gt_filepath, IMG_SIZE)
        pred_boxes = load_pred_boxes(pred_filepath, IMG_SIZE)

        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            continue  # Ignore empty frames

        num_classes = int(max(gt_boxes[:,4].max(), pred_boxes[:,4].max())) + 1
        metric_fn = MetricBuilder.build_evaluation_metric('map_2d', num_classes=num_classes)

        metric_fn.add(pred_boxes, gt_boxes)

        maps50.append(metric_fn.value(iou_thresholds=[0.5])['mAP'])
        maps5095.append(metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05))['mAP'])

    print(maps50)
    print(maps5095)

    print('mAP@0.5:', np.mean(maps50))
    print('mAP@0.5:0.95:', np.mean(maps5095))
