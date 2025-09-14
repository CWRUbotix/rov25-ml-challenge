from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
from cv2.typing import MatLike
from ultralytics import YOLO

# model = YOLO('runs/detect/train/weights/last.pt')

# paths = tuple(Path('dataset/images/val').iterdir())

# print(paths)

# results = model(paths)

# for path, result in zip(paths, results, strict=True):
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     #result.show()  # display to screen
#     result.save(filename=f'predictions/{path.stem}.jpg')  # save to disk

@dataclass
class Prediction:
    x: float
    y: float
    x2: float
    y2: float
    w: float
    h: float
    prob: float

    @staticmethod
    def fromXYWHN(xywhn: tuple[float, float, float, float], prob: float) -> 'Prediction':
        return Prediction(xywhn[0], xywhn[1],
                          xywhn[0] + xywhn[2], xywhn[1] + xywhn[3],
                          xywhn[2], xywhn[3],
                          prob)

    def to_scaled_xyxy(self, img_width: int, img_height: int) -> tuple[int, int, int, int]:
        return (
            int(self.x * img_width),
            int(self.y * img_height),
            int(self.x2 * img_width),
            int(self.y2 * img_height)
        )

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x2, self.y2)

@dataclass
class PredictionSet:
    predictions: list[Prediction]
    img: MatLike

    def annotated_img(self) -> MatLike:
        annotated = self.img.copy()
        for pred in self.predictions:
            img_height, img_width = annotated.shape[:2]
            x1, y1, x2, y2 = pred.to_scaled_xyxy(img_width, img_height)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated, f'{pred.prob * 100:.1f}%',
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f'Fish count: {len(self.predictions)}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2)
        return annotated

class Predictor:
    def __init__(self):
        self.model = YOLO('runs/detect/train/weights/last.pt')

    def predict_frame(self, img: MatLike, min_prob: float) -> PredictionSet:
        # https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-OpenCV-ONNX-Python/main.py
        blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 255, size=(640, 640),
                                     swapRB=True, ddepth=cv2.CV_32F)
        self.model.setInput(blob)
        outputs = self.model.forward()

        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes: list[tuple[float, float, float, float]] = []
        scores: list[float] = []

        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= min_prob:
                box = (
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),  # x center - width/2 = left x
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),  # y center - height/2 = top y
                    outputs[0][i][2],  # width
                    outputs[0][i][3],  # height
                )
                boxes.append(box)
                scores.append(maxScore)

        # Non-max suppression
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, min_prob, 0.45, 0.5)

        prediction_set = PredictionSet([], img)

        for box_idx in result_boxes:
            print(boxes[box_idx])  # TODO: confirm that these are normalized here
            prediction_set.predictions.append(
                Prediction.fromXYWHN(boxes[box_idx], scores[box_idx]))

        return prediction_set

    def predict_img(self, img_path: Path, min_prob: float = 0.25) -> PredictionSet:
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f'Could read image {img_path}')
        return self.predict_frame(img, min_prob)

    def predict_video(self, video_path: Path, min_prob: float = 0.25,
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
                prediction_set = self.predict_frame(frame, min_prob)
                results.append(prediction_set)
                frame_idx += 1
        cap.release()
        return results

    def annotate_img(self, input_path: Path, output_path: Path, min_prob: float = 0.25) -> None:
        prediction_set = self.predict_img(input_path, min_prob)
        result_img = prediction_set.annotated_img()
        cv2.imwrite(str(output_path), result_img)

    def annotate_video(self, video_path: Path, min_prob: float = 0.25,
                      frame_interval: int = 1) -> None:
        predictions = self.predict_video(video_path, min_prob, frame_interval)
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
    paths = tuple(Path('dataset/images/val').iterdir())
    predictor = Predictor()
    for path in paths:
        predictor.annotate_img(path, Path('output') / path.name)