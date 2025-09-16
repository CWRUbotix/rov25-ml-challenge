import torch
import cv2
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from effdet.efficientdet import HeadNet
from pathlib import Path
import sys
import torchvision.transforms as T

model_path = 'effdet.pt'
IMAGE_FOLDER = 'dataset/images/test'
OUTPUT_FOLDER = 'output/effdet_predictions'
IMG_SIZE = 768  # D2 default
NUM_CLASSES = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(model_path, num_classes):
    config = get_efficientdet_config('tf_efficientdet_d2')
    config.num_classes = num_classes
    config.image_size = (IMG_SIZE, IMG_SIZE)

    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    model = DetBenchPredict(net)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint)
    model = model.to(DEVICE)
    model.eval()
    return model

transform = T.Compose([
    T.ToTensor(),
])

def predict_and_save(model, image_path, output_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    h, w, _ = img.shape

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        preds = model(img_tensor)[0]

    with open(output_path, 'w') as f:
        for x1, y1, x2, y2, score, label in preds:
            if score < 0.01:
                continue

            # normalize to YOLO format
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            f.write(f'{0} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    model = load_model(model_path, NUM_CLASSES)
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    for img_path in Path(IMAGE_FOLDER).iterdir():
        out_file = Path(OUTPUT_FOLDER) / (img_path.stem + '.txt')
        predict_and_save(model, img_path, out_file)
        print(f'Saved predictions for {img_path.name} -> {out_file}')
