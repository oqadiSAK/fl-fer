import torch
import torch.nn as nn
import model
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils.dataset_factory import DataSetFactory
from torchsummary import summary

from utils.plot import plot_confusion_matrix, plot_loss_accuracy

device = torch.device('cpu')
shape = (44, 44)

def main():
    batch_size = 128
    lr = 0.01
    epochs = 2
    learning_rate_decay_start = 80
    learning_rate_decay_every = 5
    learning_rate_decay_rate = 0.9

    classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    network = model.Model(num_classes=len(classes)).to(device)
    summary(network, (1, shape[0], shape[1]))

    optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.9, weight_decay=5e-3)
    criterion = nn.CrossEntropyLoss()
    factory = DataSetFactory(shape)

    training_loader = DataLoader(factory.training, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = DataLoader(factory.public, batch_size=batch_size, shuffle=True, num_workers=1)

    min_validation_loss = 10000
    train_loss_values = []
    val_loss_values = []
    train_acc_values = []
    val_acc_values = []
    y_true = []
    y_pred = []
    
    for epoch in range(epochs):
        network.train()
        total = 0
        correct = 0
        total_train_loss = 0
        if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:

            #
            frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
            decay_factor = learning_rate_decay_rate ** frac
            current_lr = lr * decay_factor
            for group in optimizer.param_groups:
                group['lr'] = current_lr
        else:
            current_lr = lr

        print('learning_rate: %s' % str(current_lr))
        for i, (x_train, y_train) in enumerate(training_loader):
            optimizer.zero_grad()
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            y_predicted = network(x_train)
            loss = criterion(y_predicted, y_train)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(y_predicted.data, 1)
            total_train_loss += loss.data
            total += y_train.size(0)
            correct += predicted.eq(y_train.data).sum()
        accuracy = 100. * float(correct) / total
        print('Epoch [%d/%d] Training Loss: %.4f, Accuracy: %.4f' % (
            epoch + 1, epochs, total_train_loss / (i + 1), accuracy))

        train_loss_values.append(total_train_loss / (i + 1))
        train_acc_values.append(accuracy)   
        
        network.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            total_validation_loss = 0
            for j, (x_val, y_val) in enumerate(validation_loader):
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                y_val_predicted = network(x_val)
                val_loss = criterion(y_val_predicted, y_val)
                _, predicted = torch.max(y_val_predicted.data, 1)
                total_validation_loss += val_loss.data
                total += y_val.size(0)
                correct += predicted.eq(y_val.data).sum()
                y_true.extend(y_val.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

            accuracy = 100. * float(correct) / total
            if total_validation_loss <= min_validation_loss:
                if epoch >= 10:
                    print('saving new model')
                    state = {'net': network.state_dict()}
                    torch.save(state, 'trained/model_%d_%d.t7' % (epoch + 1, accuracy))
                min_validation_loss = total_validation_loss

            print('Epoch [%d/%d] validation Loss: %.4f, Accuracy: %.4f' % (
                epoch + 1, epochs, total_validation_loss / (j + 1), accuracy))
            
            val_loss_values.append(total_validation_loss / (j + 1))
            val_acc_values.append(accuracy)
        
    
    plot_loss_accuracy(train_loss_values, val_loss_values, train_acc_values, val_acc_values)
    plt.show()
    
    plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix for Validation Data')
    plt.show()
            


if __name__ == "__main__":
    main()