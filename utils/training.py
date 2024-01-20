import torch
import torch.nn as nn
from tqdm import tqdm

def train(model, device, train_loader, val_loader, epochs, lr, momentum, weight_decay):
    print('Training started')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loss_values = []
    val_loss_values = []
    train_acc_values = []
    val_acc_values = []

    for epoch in range(epochs):
        model.train()
        total = 0
        correct = 0
        total_train_loss = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for i, (x_train, y_train) in pbar:
            optimizer.zero_grad()
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            y_predicted = model(x_train)
            loss = criterion(y_predicted, y_train)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(y_predicted.data, 1)
            total_train_loss += loss.data.item()
            total += y_train.size(0)
            correct += predicted.eq(y_train.data).sum()

            pbar.set_description(f"Epoch {epoch+1}/{epochs}")
            pbar.set_postfix({"Training Loss": total_train_loss / (i + 1), "Accuracy": 100. * float(correct) / total})

        accuracy = 100. * float(correct) / total
        print('Epoch [%d/%d] Training Loss: %.4f, Accuracy: %.4f' % (
            epoch + 1, epochs, total_train_loss / (i + 1), accuracy))

        train_loss_values.append(total_train_loss / (i + 1))
        train_acc_values.append(accuracy)

        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            total_validation_loss = 0

            pbar_val = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
            for j, (x_val, y_val) in pbar_val:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                y_val_predicted = model(x_val)
                val_loss = criterion(y_val_predicted, y_val)
                _, predicted = torch.max(y_val_predicted.data, 1)
                total_validation_loss += val_loss.data.item()
                total += y_val.size(0)
                correct += predicted.eq(y_val.data).sum()

                pbar_val.set_description(f"Validation Epoch {epoch+1}/{epochs}")
                pbar_val.set_postfix({"Validation Loss": total_validation_loss / (j + 1), "Accuracy": 100. * float(correct) / total})

            accuracy = 100. * float(correct) / total
            print('Epoch [%d/%d] Validation Loss: %.4f, Accuracy: %.4f' % (
                epoch + 1, epochs, total_validation_loss / (j + 1), accuracy))

            val_loss_values.append(total_validation_loss / (j + 1))
            val_acc_values.append(accuracy)

    return len(train_loader.dataset), {
        "train_loss_values": train_loss_values,
        "val_loss_values": val_loss_values,
        "train_acc_values": train_acc_values,
        "val_acc_values": val_acc_values
    }

def evaluate(model, device, test_loader):
    print('Evaluating model')
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total = 0
    correct = 0
    total_test_loss = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        pbar_test = tqdm(enumerate(test_loader), total=len(test_loader), leave=False)
        for k, (x_test, y_test) in pbar_test:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            y_test_predicted = model(x_test)
            test_loss = criterion(y_test_predicted, y_test)
            _, predicted = torch.max(y_test_predicted.data, 1)
            y_true.extend(y_test.tolist())
            y_pred.extend(predicted.tolist())
            total_test_loss += test_loss.data
            total += y_test.size(0)
            correct += predicted.eq(y_test.data).sum()

            pbar_test.set_description("Evaluation")
            pbar_test.set_postfix({"Test Loss": total_test_loss / (k + 1), "Accuracy": 100. * float(correct) / total})

    accuracy = 100. * float(correct) / total
    print('Test Loss: %.4f, Accuracy: %.4f' % (total_test_loss / (k + 1), accuracy))

    return float(total_test_loss / (k + 1)), len(test_loader.dataset), {
        "test_loss": float(total_test_loss / (k + 1)),
        "accuracy": float(accuracy),
        "y_true": y_true,
        "y_pred": y_pred
    }