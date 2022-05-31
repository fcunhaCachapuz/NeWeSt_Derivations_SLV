import pandas as pd
from typing import List, Dict, Optional
from sklearn.metrics import ConfusionMatrixDisplay

# Import Plot Libraries
import matplotlib.pyplot as plt
import seaborn as sns


def plot_roc_curve(
    fpr: pd.DataFrame,
    tpr: pd.DataFrame,
    auc: float,
    algorithm: str,
    location: Optional[str] = None,
) -> None:

    """

    Function to plot the ROC curve and save it in a specific provided location.

    Parameters
    ----------
    :param fpr: The fpr metric.
    :type fpr: pd.DataFrame

    :param tpr: The tpr metric.
    :type tpr: pd.DataFrame

    :param auc: The AUC metric.
    :type auc: float

    :param algorithm: The name of the ML algorithm related to the ROC curve plot.
    :type algorithm: str

    :param location: The location to save the ROC curve plot. By default is None
    :type location: Optional[str]

    Returns
    -------
    :return: None.

    """

    if location is None:
        location = 'static/images'

    fig, ax = plt.subplots(1, 1)

    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label='ROC curve (AUC = {:.4f})'.format(auc))
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([-0.05, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic Curve')
    ax.legend(loc="lower right")
    fig.savefig('{}/{}_ROC.pdf'.format(location, algorithm))


def roc_curve_all(
    metrics: List[Dict],
    location: Optional[str] = None,
) -> None:

    """

    Function to plot the ROC curve of all tested models. Afterwards, the generated plot is saved in a specific provided location.

    Parameters
    ----------
    :param metrics: The metrics used to plot the ROC curve of each tested models. Each position of theList contains a Dictionary with fpr - pd.DataFrame (metric), tpr - pd.DataFrame (metric), auc - float (metric), model - str (name of the model).
    :type metrics: List[Dict]

    :param location: The location to save the ROC curve plot. By default is None
    :type location: Optional[str]

    Returns
    -------
    :return: None.

    """

    fig, ax = plt.subplots(1, 1)

    if location is None:
        location = 'static/images'

    configs = [
        {'color': 'tab:red', 'linestyle': 'solid'},
        {'color': 'tab:blue', 'linestyle': 'dotted'},
        {'color': 'tab:orange', 'linestyle': 'dashed'},
        {'color': 'black', 'linestyle': '0, (3, 5, 1, 5, 1, 5)'},
        {'color': 'tab:purple', 'linestyle': '0, (5, 10)'},
        {'color': 'tab:green', 'linestyle': 'dotted'},
        {'color': 'tab:gray', 'linestyle': 'solid'}
    ]

    for i, v in enumerate(metrics):
        ax.plot(
            metrics[i]['fpr'],
            metrics[i]['tpr'],
            color=configs[i]['color'],
            linestyle=configs[i]['linestyle'],
            lw=2,
            label='{} (AUC = {:.4f})'.format(metrics[i]['auc'], metrics[i]['model']))

    ax.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    ax.set_xlim([-0.05, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(b=True, linestyle='dashed')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic Curve')
    ax.legend(loc="lower right")
    fig.savefig('{}/ALL_ROC.pdf'.format(location))


def plot_confusion_matrix(
    y_test: pd.DataFrame,
    y_pred: pd.DataFrame,
    model_name: str,
    location: Optional[str] = None,
) -> None:

    """

    Function to plot the confusion matrix from model predictions.

    Parameters
    ----------
    :param y_test: The labels from the test set.
    :type y_test: pd.DataFrame

    :param y_pred: The predicted labels.
    :type y_pred: pd.DataFrame

    :param model_name: The name of the ML model.
    :type model_name: str

    :param location: The location to save the confusion matrix plot. By default is None
    :type location: Optional[str] = None

    Returns
    -------
    :return: None.

    """

    if location is None:
        location = 'static/images'

    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    disp.plot()
    plt.savefig('{}/Confusion_Matrix_{}'.format(location, model_name))
