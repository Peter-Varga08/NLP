import wandb


def log_results(results):
    wandb.log(results)


def plot_precision_recall(y_true, y_pred, labels):
    wandb.sklearn.plot_precision_recall(y_true, y_pred, labels)


def plot_feature_importances(model, feature_names):
    wandb.sklearn.plot_feature_importances(model, feature_names)
