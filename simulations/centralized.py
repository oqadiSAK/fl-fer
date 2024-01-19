import torch
import argparse
from utils.training import train, evaluate
from utils.plot import plot_confusion_matrix, plot_loss_accuracy
from utils.loaders import load_data_loaders, load_model, CLASSES

def main():
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-3, help='Weight decay')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device)
    training_loader, test_loader, validation_loader = load_data_loaders()
    _, train_metrics = train(model, device, training_loader, validation_loader, 
          args.epochs, args.learning_rate, args.momentum, args.weight_decay)
    plot_loss_accuracy(train_metrics, filename="centralized_loss_accuracy_plot")
    
    _, _, test_metrics = evaluate(model, device, test_loader)
    plot_confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"], CLASSES, filename="centralized_confusion_matrix")
    accuracy = int(test_metrics["accuracy"])
    torch.save(model.state_dict(), f'trained/centralized_model_{accuracy}.t7')

if __name__ == "__main__":
    main()