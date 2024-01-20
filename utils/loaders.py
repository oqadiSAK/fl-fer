from model import Model
from torch.utils.data import DataLoader
from utils.dataset_factory import DataSetFactory

CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def load_data_loaders(batch_size=128, num_workers=0):
    factory = DataSetFactory()
    training_loader = DataLoader(factory.training, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(factory.public, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(factory.private, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return training_loader, test_loader, validation_loader

def load_dynamic_train_loader(batch_size=128, num_workers=0):
    try:
        dynamic_dataset = DataSetFactory.get_dynamic_dataset()
        training_loader = DataLoader(dynamic_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return training_loader
    except ValueError as e:
        print(e)
        raise

def load_model(device):
    model = Model(num_classes=len(CLASSES)).to(device)
    return model