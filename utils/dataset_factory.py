import torchvision.transforms as transforms
import numpy as np
import csv
from PIL import Image
from torchvision.transforms import ToTensor
from utils.dataset import DataSet

class DataSetFactory:
    SHAPE = (44, 44)
    
    TRAIN_TRANSFORM = transforms.Compose([
        transforms.RandomCrop(SHAPE[0]), 
        transforms.RandomHorizontalFlip(),
        ToTensor(),
    ])

    VAL_TRANSFORM = transforms.Compose([
        transforms.CenterCrop(SHAPE[0]),  
        ToTensor(),
    ])
    
    def __init__(self):
        images = []
        emotions = []
        private_images = []
        private_emotions = []
        public_images = []
        public_emotions = []

        with open('data/fer2013.csv', 'r') as csvin:
            data = csv.reader(csvin)
            next(data)
            for row in data:
                face = [int(pixel) for pixel in row[1].split()]
                face = np.asarray(face).reshape(48, 48)
                face = face.astype('uint8')

                if row[-1] == 'Training':
                    emotions.append(int(row[0]))
                    images.append(Image.fromarray(face))
                elif row[-1] == "PrivateTest":
                    private_emotions.append(int(row[0]))
                    private_images.append(Image.fromarray(face))
                elif row[-1] == "PublicTest":
                    public_emotions.append(int(row[0]))
                    public_images.append(Image.fromarray(face))

        print('Training size %d : Validation size %d : Test size %d' % (
            len(images), len(private_images), len(public_images)))

        self.training = DataSet(transform=DataSetFactory.TRAIN_TRANSFORM, images=images, emotions=emotions)
        self.private = DataSet(transform=DataSetFactory.VAL_TRANSFORM, images=private_images, emotions=private_emotions)
        self.public = DataSet(transform=DataSetFactory.VAL_TRANSFORM, images=public_images, emotions=public_emotions)

    
    @staticmethod
    def get_dynamic_dataset(threshold=5):
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
            open('gui/local/local_saves.csv', 'w').close()

        dynamic_dataset = DataSet(transform=DataSetFactory.TRAIN_TRANSFORM, images=images, emotions=emotions)
        return dynamic_dataset