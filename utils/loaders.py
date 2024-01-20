import torchvision.transforms as transforms
import numpy as np
import csv
from PIL import Image
from utils.dataset import DataSet
from model import Model
from torch.utils.data import DataLoader
from torch.utils.data import Subset

CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
_SHAPE = (44, 44)

_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(_SHAPE[0]), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

_VAL_TRANSFORM = transforms.Compose([
    transforms.CenterCrop(_SHAPE[0]),  
    transforms.ToTensor(),
])

def load_data_loaders(batch_size=128, num_workers=0):
    training_dataset, test_dataset, validation_dataset = _get_datasets()
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return training_loader, test_loader, validation_loader

def load_train_loader(batch_size=128, num_workers=0):
    training_dataset = _get_training_dataset()
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return training_loader

def load_test_loader(batch_size=128, num_workers=0):
    test_dataset = _get_test_dataset()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return test_loader

def load_test_loader_random_part(batch_size=128, num_workers=0, percentage=0.2):
    test_dataset = _get_test_dataset()
    num_samples = round(len(test_dataset) * percentage)
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    test_dataset_part = Subset(test_dataset, indices)
    test_loader = DataLoader(test_dataset_part, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return test_loader

def load_validate_loader(batch_size=128, num_workers=0):
    validation_dataset = _get_validation_dataset()
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return validation_loader

def load_dynamic_train_loader(batch_size=128, num_workers=0):
    try:
        dynamic_dataset = _get_dynamic_dataset()
        training_loader = DataLoader(dynamic_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return training_loader
    except ValueError:
        raise

def load_model(device):
    model = Model(num_classes=len(CLASSES)).to(device)
    return model

def _get_datasets():
    return _get_training_dataset(), _get_test_dataset(), _get_validation_dataset()
    
def _get_training_dataset():
    return _get_dataset('data/train.csv', _TRAIN_TRANSFORM)

def _get_test_dataset():
    return _get_dataset('data/test.csv', _VAL_TRANSFORM)

def _get_validation_dataset():
    return _get_dataset('data/validate.csv', _VAL_TRANSFORM)

def _get_dynamic_dataset(threshold=5):
    images = []
    emotions = []
    with open('gui/local/local_saves.csv', 'r') as csvin:
        data = csv.reader(csvin)
        next(data)
        for row in data:
            face = [int(pixel) for pixel in row[1].split()]
            face = np.asarray(face).reshape(48, 48)
            face = face.astype('uint8')
            emotions.append(int(row[0]))
            images.append(Image.fromarray(face))

    if len(images) < threshold:
        raise ValueError(f"Number of images is less than the threshold value {threshold}")
    else:
        with open('gui/local/local_saves.csv', 'w') as file:
            file.write('emotion,pixels\n')

    dynamic_dataset = DataSet(transform=_TRAIN_TRANSFORM, images=images, emotions=emotions)
    return dynamic_dataset

def _get_dataset(file_path, transform):
    images = []
    emotions = []

    with open(file_path, 'r') as csvin:
        data = csv.reader(csvin)
        next(data)
        for row in data:
            face = [int(pixel) for pixel in row[1].split()]
            face = np.asarray(face).reshape(48, 48)
            face = face.astype('uint8')

            emotions.append(int(row[0]))
            images.append(Image.fromarray(face))

    return DataSet(transform=transform, images=images, emotions=emotions)