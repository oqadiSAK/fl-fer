def map_train_metrics(train_metrics):
    metrics = ["train_loss_values", "val_loss_values", "train_acc_values", "val_acc_values"]
    fl_metrics = {metric[:-7]: train_metrics.get(metric, [None])[-1] for metric in metrics}
    return fl_metrics

def map_eval_metrics(eval_metrics):
    keys_to_remove = ['y_true', 'y_pred', 'test_loss']
    for key in keys_to_remove:
        eval_metrics.pop(key, None)
    return eval_metrics