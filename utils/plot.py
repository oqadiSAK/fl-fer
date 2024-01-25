import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from pathlib import Path

class Plotter:
    def __init__(self, output_dir="output/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_confusion_matrix(self, y_true, y_pred, classes, filename, title='Confusion matrix', cmap=plt.cm.Blues):
        cm = confusion_matrix(y_true, y_pred)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        self._save_plot(filename)

    def plot_loss_accuracy(self, train_metrics, filename):
        train_loss = train_metrics["train_loss_values"]
        val_loss = train_metrics["val_loss_values"]
        train_acc = train_metrics["train_acc_values"]
        val_acc = train_metrics["val_acc_values"]

        epochs = range(1, len(train_loss) + 1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_acc, 'r', label='Training accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        self._save_plot(filename)

    def plot_fl_losses(self, history, filename="fl_loss_plot"):
        if history.losses_distributed:
            rounds, distributed_losses = zip(*history.losses_distributed)
            plt.plot(rounds, distributed_losses, label="Distributed Loss")

        if history.losses_centralized:
            rounds, centralized_losses = zip(*history.losses_centralized)
            plt.plot(rounds, centralized_losses, label="Centralized Loss")

        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.title("Federated Learning Loss Curve")
        plt.legend()
        self._save_plot(filename)

    def plot_fl_distributed_fit_metrics(self, history, filename="fl_metrics_fit_plot"):
        self._plot_fl_metrics(history.metrics_distributed_fit, "Distributed Fit Metrics", filename)

    def plot_fl_distributed_evaluation_metrics(self, history, filename="fl_metrics_eval_plot"):
        self._plot_fl_metrics(history.metrics_distributed, "Distributed Evaluation Metrics", filename)

    def plot_fl_centralized_metrics(self, history, filename="fl_metrics_centralized_plot"):
        self._plot_fl_metrics(history.metrics_centralized, "Centralized Metrics", filename)
    
    def _plot_fl_metrics(self, metrics_dict, title, filename):
        for metric_name, values in metrics_dict.items():
            rounds, metric_values = zip(*values)
            plt.plot(rounds, metric_values, label=metric_name)

        plt.xlabel("Round")
        plt.ylabel("Metric Value")
        plt.title(title)
        plt.legend()
        self._save_plot(filename)
        
    def _save_plot(self, filename):
        plt.savefig(self.output_dir / f"{filename}.png")
        plt.close()