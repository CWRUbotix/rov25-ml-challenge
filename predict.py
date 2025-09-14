from pathlib import Path
from dataclasses import dataclass
import argparse

import cv2
import numpy as np
from cv2.typing import MatLike
from ultralytics import YOLO
from ultralytics.utils.nms import TorchNMS

NETWORK_IMG_SHAPE = (640, 640)
REAL_IMG_SHAPE = (1920, 1080)

@dataclass
class Prediction:
    x1: float
    y1: float
    x2: float
    y2: float
    w: float
    h: float
    score: float

    @staticmethod
    def fromXYWHN(xywhn: tuple[float, float, float, float], score: float) -> 'Prediction':
        wr = xywhn[2] / 2
        hr = xywhn[3] / 2
        return Prediction(xywhn[0] - wr, xywhn[1] - hr,
                          xywhn[0] + wr, xywhn[1] + hr,
                          xywhn[2], xywhn[3],
                          score)

    def to_scaled_xyxy(self, img_width: int, img_height: int) -> tuple[int, int, int, int]:
        return (
            int(self.x1 * img_width),
            int(self.y1 * img_height),
            int(self.x2 * img_width),
            int(self.y2 * img_height)
        )

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y2, self.x2, self.y2)

@dataclass
class PredictionSet:
    predictions: list[Prediction]
    img: MatLike

    def annotated_img(self, resize_shape_wh: tuple[int, int] | None = None) -> MatLike:
        annotated = self.img.copy()
        if resize_shape_wh is not None:
            annotated = cv2.resize(annotated, resize_shape_wh,
                                   interpolation=cv2.INTER_LINEAR)
        for pred in self.predictions:
            img_height, img_width = annotated.shape[:2]
            x1, y1, x2, y2 = pred.to_scaled_xyxy(img_width, img_height)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated, f'{pred.score * 100:.1f}%',
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f'Fish count: {len(self.predictions)}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2)
        return annotated

class Predictor:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict_frame(self, img: MatLike, min_score: float,
                      iou_threshold: float = 0.3) -> PredictionSet:
        results = self.model(img)

        xyxy = results[0].boxes.xyxy
        xywhn = results[0].boxes.xywhn
        scores = results[0].boxes.conf
        nms_indices = TorchNMS.nms(
            boxes=xyxy,
            scores=scores,
            iou_threshold=iou_threshold,
        )

        prediction_set = PredictionSet([], img)

        for box, score in zip(xywhn[nms_indices], scores[nms_indices], strict=True):
            prediction_set.predictions.append(Prediction.fromXYWHN(box, score))

        return prediction_set

    def predict_img(self, img_path: Path, min_score: float = 0.25) -> PredictionSet:
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f'Could read image {img_path}')
        return self.predict_frame(img, min_score)

    def predict_video(self, video_path: Path, min_score: float = 0.25,
                      frame_interval: int = 1) -> list[PredictionSet]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f'Could read video {video_path}')
        results = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                frame.resize(NETWORK_IMG_SHAPE)
                prediction_set = self.predict_frame(frame, min_score)
                results.append(prediction_set)
                frame_idx += 1
        cap.release()
        return results

    def annotate_img(self, input_path: Path, output_path: Path, min_score: float = 0.25) -> None:
        prediction_set = self.predict_img(input_path, min_score)
        result_img = prediction_set.annotated_img(REAL_IMG_SHAPE)
        cv2.imwrite(str(output_path), result_img)

    def annotate_video(self, video_path: Path, min_score: float = 0.25,
                      frame_interval: int = 1) -> None:
        predictions = self.predict_video(video_path, min_score, frame_interval)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f'Could read video {video_path}')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('annotated_' + video_path.name, fourcc, fps, (width, height))
        cap.release()

        for prediction_set in predictions:
            annotated = prediction_set.annotated_img()
            for _ in range(frame_interval):
                out.write(annotated)
        out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='predict.py',
                                     description='Make predictions with a Yolo model')

    parser.add_argument('-m', '--model-path', required=True,
                        help='Path to model .pt file (i.e. runs/detect/train/weights/last.pt)',
                        default='runs/detect/train/weights/last.pt')

    parsed_args = parser.parse_args()

    paths = tuple(Path('dataset/images/val').iterdir())
    predictor = Predictor(parsed_args.model_path)
    Path('output').mkdir(exist_ok=True)
    for path in paths:
        predictor.annotate_img(path, Path('output') / path.name)