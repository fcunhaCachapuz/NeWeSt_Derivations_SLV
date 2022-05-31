import pandas as pd
from typing import Dict
import time

# Bayesian optimization (Hyperparameter tuning)
from hyperopt import STATUS_OK

# Metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve

# Classification Models
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def number_iterations_rolling_window(
    D: pd.DataFrame,
    W: int,
    T: int,
    S: int
) -> int:

    """

    Function to calculate the number of iterations regarding the rolling window mechanism,
    using the following formula:  U = (D-(W+T))/S

    Parameters
    ----------
    :param D: The Dataframe of Weighings.
    :type D: pd.DataFrame

    :param W: The training data size.
    :type W: int

    :param T: The test data size.
    :type T: int

    :param S: The sliding (jump/slide at each iteration).
    :type S: int

    Returns
    -------
    :return: Dataframe with given value replaced.
    :rtype: pd.DataFrame

    """

    D = len(D)
    U = (D - (W + T)) / S
    return int(U)


def rolling_mechanism(
    df_size: int,
    window: int,
    increment: int,
    iteration: int,
    sliding: int
) -> Dict:

    """

    Function to implement the rolling window mechanism (calculate the start and end of training and test size)

    Parameters
    ----------
    :param df_size: The Dataframe size.
    :type df_size: int

    :param window: The Dataframe size.
    :type window: int
    
    :param increment: The test data size
    :type increment: int

    :param iteration: The current rolling window iteraction
    :type iteration: int

    :param sliding: The sliding (jump/slide at each iteration)
    :type sliding: int

    Returns
    -------~
    :return: Dataframe with given value replaced.
    :rtype: pd.DataFrame

    """

    # Calculate the start and end of W (training data size)
    end_train = window + increment * (iteration - 1)
    end_train = min(end_train, df_size)
    start_train = max((end_train - window + 1), 1)
    TR = [start_train, end_train]

    # Calculate the start and end of T (test data size)
    end_test = end_train + sliding
    end_test = min(end_test, df_size)
    start_test = end_train + 1
    if start_test < end_test:
        TS = [start_test, end_test]
    else:
        TS = None

    return {"tr": TR, "ts": TS}


def get_train_test_set(
    rolling: Dict,
    X: pd.DataFrame,
    y: pd.DataFrame
) -> Dict:

    """

    Function to get train and test set, as well the corresponding labels

    Parameters
    ----------
    :param rolling: The rolling window mechanism (window size for train and test set).
    :type rolling: Dict

    :param X: The training Dataframe.
    :type X: pd.DataFrame

    :param y: The testing Dataframe.
    :type y: pd.DataFrame

    Returns
    -------
    :return: X_train (Training set), y_train (Labels of training set), X_test (Testing set) and y_test (Labels of testing set).
    :rtype: Dict

    """

    X_train = X.iloc[rolling["tr"][0]: rolling["tr"][1]]
    y_train = y.iloc[rolling["tr"][0]: rolling["tr"][1]]
    X_test = X.iloc[rolling["ts"][0]: rolling["ts"][1]]
    y_test = y.iloc[rolling["ts"][0]: rolling["ts"][1]]

    return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}


def fit_model_hyperopt(
    params: Dict,
    estimator: str,
    kfolds: int,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame
) -> Dict:

    """

    Function to perform hyperparameter tunning using training data

    Parameters
    ----------
    :param params: The selected hyperameter values from the search space.
    :type params: Dict

    :param estimator: The name of classification estimator.
    :type estimator: str

    :param kfolds: The number of folds of cross-validation.
    :type kfolds: int

    :param X_train: The training Dataframe.
    :type X_train: pd.DataFrame

    :param y_train: The labels from the training set.
    :type y_train: pd.DataFrame

    Returns
    -------
    :return: Loss (value to be minimized) and status.
    :rtype: Dict
        
    """

    if estimator == "RF":
        # Random Forest Classifier
        print(
            "Random Forest -> max_depth: {0} - min_samples_split: {1}".format(
                int(params['max_depth']), int(params['min_samples_split'])))

        model = RandomForestClassifier(
            n_estimators=200, max_depth=int(params['max_depth']), min_samples_split=int(params['min_samples_split']))
    elif estimator == "DT":
        # Decision Tree Classifier
        print(
            "Decision Tree Classifier -> max_depth: {0}".format(int(params['max_depth'])))

        model = DecisionTreeClassifier(max_depth=int(params['max_depth']))
    elif estimator == "LR":
        # Logistic Regression
        print("Logistic Regression -> regParam: {0} | l1_ratio: {1}".format(
            float(params['C']), float(params['l1_ratio'])))

        model = LogisticRegression(max_iter=100, C=float(
            params['C']), penalty='elasticnet', l1_ratio=float(params['l1_ratio']), solver='saga')
    elif estimator == "GBT":
        # Gradient-Boosted Tree Classifier
        print(
            "Gradient-Boosted Tree Classifier -> max_depth: {0}".format(int(params['max_depth'])))

        model = GradientBoostingClassifier(n_estimators=200, max_iter=100, max_depth=int(
            params['max_depth']))
    elif estimator == "SVC":
        # Linear Support Vector Machine
        print("Support Vector Machine -> C: {0}".format(float(params['C'])))
        model = SVC(max_iter=100, C=float(params['C']))
    elif estimator == "MLP":
        # Multilayer Perceptron Classifier

        print(
            "Multilayer Perceptron -> max_iter: {0} - alpha: {1}".format(
                int(params['max_iter']), int(params['alpha'])))

        inputLayer = len(X_train.columns)
        hiddenLayer = int(round(inputLayer / 2))

        print("Input Layer: {}".format(inputLayer))

        hidden_layers = (hiddenLayer,)

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers, max_iter=100, alpha=int(params['alpha']), solver='lbfgs')

    cval = cross_val_score(model, X_train, y_train,
                           scoring='roc_auc', cv=kfolds)

    auc = cval.mean()
    # Because fmin() tries to minimize the objective, this function must return the negative auc.
    return {'loss': -auc, 'status': STATUS_OK}


def fit_model_h(
    params: Dict,
    estimator: str,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Dict:

    """

    Function to train models using training data and predict using testing data

    Parameters
    ----------
    :param params: The selected hyperameter values from the search space.
    :type params: Dict

    :param estimator: The name of classification estimator.
    :type estimator: str

    :param X_train: The training DataFrame.
    :type X_train: pd.DataFrame

    :param y_train: The labels from the training set.
    :type y_train: pd.DataFrame

    :param X_test: The test DataFrame.
    :type X_test: pd.DataFrame

    Returns
    -------
    :return: The predicted values, model fitted, elapsed time to feed models and elapsed time to predict
    :rtype: Dict
    
    """

    if estimator == "RF" or estimator == "DT" or estimator == "GBT":
        search_space = {
            'max_depth': [2, 5, 10, 20, 30],
            'min_samples_split': [2, 6, 10]
        }

        best_maxDepth = int(params['max_depth'])
        best_minSplit = int(params['min_samples_split'])

        Model_maxDepth = search_space['max_depth'][best_maxDepth]
        Model_minSplit = search_space['min_samples_split'][best_minSplit]

        if estimator == "RF":
            # Random Forest Classifier
            model = RandomForestClassifier(
                n_estimators=200, max_depth=Model_maxDepth, min_samples_split=Model_minSplit)

        elif estimator == "DT":
            # Decision Tree Classifier
            model = DecisionTreeClassifier(
                max_depth=Model_maxDepth, min_samples_split=Model_minSplit)

        elif estimator == "GBT":
            model = GradientBoostingClassifier(
                n_estimators=200, max_iter=100, max_depth=Model_maxDepth, min_samples_split=Model_minSplit)

    elif estimator == "LR" or estimator == "SVC":
        search_space1 = {'C': [0.01, 0.1, 0.5, 1.0, 2.0],
                         'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]}

        best_C = int(params['C'])
        Model_C = search_space1['C'][best_C]
        if estimator == "LR":
            best_l1_ratio = int(params['l1_ratio'])
            Model_l1_ratio = search_space1['l1_ratio'][best_l1_ratio]

            # Logistic Regression
            model = LogisticRegression(penalty='elasticnet',
                                       max_iter=100, C=Model_C, l1_ratio=Model_l1_ratio, solver='saga')

        elif estimator == "SVC":
            # Support Vector Machine
            model = SVC(max_iter=100, C=Model_C)

    elif estimator == "MLP":
        # Multilayer Perceptron Classifier
        inputLayer = len(X_train.columns)
        hiddenLayer = int(round(inputLayer / 2))

        print("Input Layer: {}".format(inputLayer))

        hidden_layers = (hiddenLayer,)

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers, max_iter=100, solver='lbfgs')

    start_time = time.time()
    fitmodel = model.fit(X_train, y_train)
    end_time = time.time()
    time_elapsed1 = (end_time - start_time)

    start_time = time.time()
    results = fitmodel.predict(X_test)
    end_time = time.time()
    time_elapsed2 = (end_time - start_time)

    return {'y_pred': results, 'model': fitmodel, 'time_train': time_elapsed1, 'time_predict': time_elapsed2}


def model_evaluation_auc(
    y_test: pd.DataFrame,
    y_pred: pd.DataFrame
) -> float:

    """

    Function to calculate the AUC evaluation metric

    Parameters
    ----------
    :param y_test: The labels from the test set.
    :type y_test: pd.DataFrame

    :param y_pred: The predicted labels.
    :type y_pred: pd.DataFrame

    Returns
    -------
    :return: The AUC metric.
    :rtype: float

    """

    auc = roc_auc_score(y_test, y_pred)

    return auc


def get_roc_curve_metrics(
    y_test: pd.DataFrame,
    y_pred: pd.DataFrame
) -> Dict:

    """
    Function to calculate the AUC evaluation metric

    Parameters
    ----------
    :param y_test: The labels from the test set.
    :type y_test: pd.DataFrame

    :param y_pred: The predicted labels.
    :type y_pred: pd.DataFrame

    Returns
    -------
    :return: The fpr and tpr metrics, and thresholds.
    :rtype: Dict
    
    """

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    return {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
