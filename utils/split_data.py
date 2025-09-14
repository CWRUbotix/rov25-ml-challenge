from sklearn.model_selection import train_test_split
from pathlib import Path
import shutil

SEED = 10

def split_data(imgs_path: Path, labels_path: Path, output_path: Path, val_size=0.2, test_size=0.2):
    images = tuple(imgs_path.iterdir())
    train_images, test_images = train_test_split(images, test_size=test_size, random_state=SEED)
    train_images, val_images = train_test_split(train_images, test_size=val_size, random_state=SEED)

    for subset, imgs_subset in (('train', train_images), ('val', val_images), ('test', test_images)):
        output_imgs_path = output_path / 'images' / subset
        output_labels_path = output_path / 'labels' / subset
        output_imgs_path.mkdir(parents=True, exist_ok=True)
        output_labels_path.mkdir(parents=True, exist_ok=True)
        for img_path in imgs_subset:
            shutil.copy(imgs_path / img_path.name, output_imgs_path / img_path.name)
            label_file_name = img_path.stem + '.txt'
            shutil.copy(labels_path / label_file_name, output_labels_path / label_file_name)

if __name__ == '__main__':
    split_data(
        Path('dataset') / 'raw' / 'images',
        Path('dataset') / 'raw' / 'labels',
        Path('dataset'),
    )