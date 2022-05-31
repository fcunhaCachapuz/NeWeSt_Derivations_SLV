import statistics
import pandas as pd
from typing import Dict
from functools import partial
# Generate and plot a synthetic imbalanced classification dataset
from collections import Counter
from imblearn.over_sampling import SMOTE
from hyperopt import fmin, tpe, hp, Trials

import data_ingestion as d_ingest
import data_preparation as d_prep
import data_visualization as d_visual
import modeling_evaluation as mod_eval
import extra_functions as extra_func
from statsmodels.stats.outliers_influence import variance_inflation_factor


def data_preparation(
    df: pd.DataFrame
) -> pd.DataFrame:

    """

    Function to perform data preparation tasks

    Parameters
    -------
    :param df: The Dataframe of Weighings.
    :type df: pd.DataFrame
    
    Returns
    -------
    :return: Dataframe transformed.
    :rtype: pd.DataFrame.

    """

    # Columns to be dropped (unrequired columns)
    columns_drop = [
        'TipoDoc', 'estado', 'bruto', 'Liquido', 'DataCriacao',
        'Dataentrada', 'DataInicioOperacao', 'DataFimOperacao',
        'BrutoData', 'DataFecho', 'CodEntidade', 'CodMotorista', 'NomeMotorista'
    ]

    # Columns to be renamed
    columns_rename = {
        'PostoOperacao': 'Station',
        'TipoViatura': 'Vehicle_Type',
        'DescProduto': 'Product',
        'CodProduto': 'CodProduct',
        'Tara': 'Tare',
        'Matricula': 'Plate',
        'qtdpedida': 'Qty_Ordered',
        'TaraData': 'Tare_Date',
        'percDiff': 'Deviation'
    }

    # Remove special character from column names
    df.columns = df.columns.str.replace('[^A-Za-z0-9]+', '', regex=True)

    # Filter all rows in which the net weighing is less than or equal to zero
    df = df.drop(df[df['Liquido'] <= 0].index)

    # Remove/Dropp unrequired columns
    df = d_prep.drop_columns(df, columns_drop)

    # Rename the columns
    df.rename(columns=columns_rename, inplace=True)

    # Replace ',' for '.' in order to afterwards convert Deviation to float
    df = d_prep.replace_column_value(df, 'Deviation', ',', '.')

    # Transform TaraData to Datetime format type
    df = d_prep.date_transformation(df, 'Tare_Date')

    # Fillna in the PostoOperacao attribute by "Unknown"
    df = d_prep.dataframe_fillna(df, 'Station', 'Unknown')

    # Convert Qty_Ordered and Deviation attributes to float format type
    df[['Qty_Ordered', 'Deviation']] = df[[
        'Qty_Ordered', 'Deviation']].astype(float)

    # categorical_columns = ['Vehicle_Type', 'Station']

    # df = d_prep.one_hot_encoding_categorical(df, categorical_columns)

    return df


def feature_engineering(
    df: pd.DataFrame
) -> pd.DataFrame:

    """

    Function to perform data preparation tasks

    Parameters
    -------
    :param df: The Dataframe of Weighings.
    :type df: pd.DataFrame
    
    Returns
    -------
    :return: The processed DataFrame.
    :rtype: pd.DataFrame.

    """

    # Create new Dataframe columns from the datatime attributes (e.g., Hour, Day, Month)
    df = d_prep.get_datetime_attributes(df, 'Tare_Date')

    # Defining Supervised and Unsupervised Work Shifts
    df["Inspection"] = df['Hour'].apply(
        lambda x: 0 if x >= 6 and x < 18 else 1)

    # Create Labels for classification
    df["Block"] = df['Deviation'].apply(
        lambda x: 0 if x >= -2 and x <= 2 else 1)

    # Sort Dataframe by Tare date
    df = df.sort_values(by="Tare_Date")

    # Create new features from existing dataset variables
    # df = d_prep.rolling_window_mean(df, ['Deviation', 'Average_Deviation_Vehicle_W5', 'Vehicle_Type'], 5, 1)

    df = d_prep.rolling_window_mean(
        df, ['Deviation', 'Average_Deviation_Station_W5', 'Station'], 5, 1)

    # df = d_prep.weekly_daily_average(df, ['Deviation', 'Average_Vehicle_Weekly', 'Tare_Date', 'Vehicle_Type'], 'W')

    df = d_prep.weekly_daily_average(
        df, ['Deviation', 'Average_Station_Weekly', 'Tare_Date', 'Station'], 'W')

    # df = d_prep.weekly_daily_average(df, ['Deviation', 'Average_Vehicle_Hourly', 'Tare_Date', 'Vehicle_Type'], 'H')

    df = d_prep.weekly_daily_average(
        df, ['Deviation', 'Average_Station_Hourly', 'Tare_Date', 'Station'], 'H')

    # Drop all NaN from dataset
    df = df.dropna()
    df = df.reset_index(drop=True)

    df = d_prep.rolling_window_mean(
        df, ['Block', 'Percentage_Blocks', 'Plate'], 5, 1)

    df['Percentage_Blocks'] = df['Percentage_Blocks'].fillna(0)

    return df


def vif_calculation(
    df: pd.DataFrame
) -> pd.DataFrame:

    """

    Function to calculate the Multicollinearity between the selected features
    using Variable Inflation Factors (VIF), in order to determines the strength
    of the correlation of a variable     with a group of other independent variables
    in a dataset.
    OBS: VIF starts usually at 1 and anywhere exceeding 10 indicates
    high multicollinearity between the independent variables.

    Parameters
    -------
    :param df: The Dataframe of Weighings.
    :type df: pd.DataFrame
    
    Returns
    -------
    :return: VIF Dataframe with the name of each features and the corresponding VIF.
    :rtype: pd.DataFrame.

    """

    vif = pd.DataFrame()
    vif["Features"] = df.columns
    vif["VIF"] = [variance_inflation_factor(
        df.values, i) for i in range(df.shape[1])]

    return vif


def modeling(
    df: pd.DataFrame
) -> Dict:


    """

    Function to perform modeling and evaluation tasks

    Parameters
    -------
    :param df: The Dataframe of Weighings.
    :type df: pd.DataFrame

    
    Returns
    -------
    :return:  Lists of AUC for each tested models regarding each Rolling Window (RW) iterations.
    :rtype: pd.DataFrame.

    """

    df = df.drop([
        "Vehicle_Type",
        "CodProduct",
        "Product",
        "Deviation",
        "Station",
        "Tare_Date",
        "Plate",
        "Tare"
    ], axis=1)

    # Separate the label from dataframe
    X = df.drop("Block", axis=1)
    y = df["Block"]

    # global scaler_model
    kfolds = 3
    n_iter = 5

    vif = vif_calculation(X)
    print("Features Multicollinearity:")
    print("{}".format(vif))

    # Possible values of parameters (Search space of parameters)
    search_space1 = {
        'max_depth': hp.choice('max_depth', [2, 5, 10, 20, 30]),
        'min_samples_split': hp.choice('min_samples_split', [2, 6, 10])
    }

    search_space2 = {
        'C': hp.choice('C', [0.01, 0.1, 0.5, 1.0, 2.0]),
        'l1_ratio': hp.choice('l1_ratio', [0.0, 0.25, 0.5, 0.75, 1.0])
    }

    search_space3 = {
        'C': hp.choice('C', [0.01, 0.1, 0.5, 1.0, 2.0])
    }

    search_space4 = {
        'max_iter': hp.choice('max_iter', [100]),
        'alpha': hp.choice('alpha', [0.0001, 0.05])
    }

    models = {
        "RF": ['Random Forest', search_space1],
        "DT": ['Decision Tree', search_space1],
        # "GBT": ['Gradient Boosted Tree', search_space1],
        "LR": ['Logistic Regression', search_space2],
        # "SVC": ['Support Vector Machine', search_space3],
        # "MLP": ['Multilayer Perceptron', search_space4]
    }

    dict_auc = {}
    dict_time_train = {}
    dict_time_predict = {}

    T = 635
    S = T
    W = 31500
    iterations = mod_eval.number_iterations_rolling_window(df, W, T, S)
    print("Total: {} -> Iterations: {}".format(len(df), iterations))
    sm = SMOTE()
    df_size = len(df)

    for i in range(0, iterations):
        rolling = mod_eval.rolling_mechanism(df_size, W, T, i + 1, S)

        print(
            "Train Range: {} - Test Range: {}".format(rolling['tr'], rolling['ts']))

        data_split = mod_eval.get_train_test_set(rolling, X, y)

        X_train = data_split['X_train']
        y_train = data_split['y_train']
        X_test = data_split['X_test']
        y_test = data_split['y_test']

        X_train, y_train = sm.fit_resample(X_train, y_train)

        if i == 0:
            scaler = d_prep.standardization(X_train, X_test)
            scaler_model = scaler[0]
            X_train_scaled = scaler[1]
            X_test_scaled = scaler[2]

            for k, v in models.items():
                list_auc = []
                list_time_train = []
                list_time_predict = []

                print("\n{} Hyperparameter Optimization...".format(
                    models[k][0]))
                trials = Trials()

                best_Parameters = fmin(
                    fn=partial(
                        mod_eval.fit_model_hyperopt,
                        estimator=k,
                        kfolds=kfolds,
                        X_train=X_train_scaled,
                        y_train=y_train
                    ),  # function to optimize
                    space=models[k][1],  # Defines space of hyperparameters
                    # optimization algorithm, hyperotp will select its
                    # parameters automatically (Search algorithm: Tree of Parzen Estimators, a Bayesian method)
                    algo=tpe.suggest,
                    max_evals=n_iter,  # maximum number of iterations
                    trials=trials  # logging
                )

                print("\nFit {}...".format(models[k][0]))
                results = mod_eval.fit_model_h(
                    params=best_Parameters,
                    estimator=k,
                    X_train=X_train_scaled,
                    y_train=y_train,
                    X_test=X_test_scaled
                )
                predictions = results['y_pred']
                model = results['model']
                time_elapsed1 = results['time_train']
                time_elapsed2 = results['time_predict']

                areaUnderROC = mod_eval.model_evaluation_auc(
                    y_test, predictions)

                optResult = trials.results

                print("################# Iteration {} ###################".format(i + 1))
                print("Best params {}".format(best_Parameters))
                print("Hyperparameter Optimization results: {}".format(optResult))
                print(
                    "{}: Time-Elapsed Fit - '{}'".format(models[k][0], time_elapsed1))
                print(
                    "{}: Time-Elapsed Transform - '{}'".format(models[k][0], time_elapsed2))
                print(
                    "{}: Area under ROC - '{}'".format(models[k][0], areaUnderROC))
                list_auc.append(areaUnderROC)
                list_time_train.append(time_elapsed1)
                list_time_predict.append(time_elapsed2)
                dict_auc[k] = list_auc
                dict_time_train[k] = list_time_train
                dict_time_predict[k] = list_time_predict
        else:
            X_train_scaled = scaler_model.transform(X_train)
            X_test_scaled = scaler_model.transform(X_test)

            for k, v in models.items():
                list_auc = []
                print("\n{} Hyperparameter Optimization...".format(
                    models[k][0]))
                trials = Trials()

                best_Parameters = fmin(
                    fn=partial(
                        mod_eval.fit_model_hyperopt,
                        estimator=k,
                        kfolds=kfolds,
                        X_train=X_train_scaled,
                        y_train=y_train
                    ),  # function to optimize
                    space=models[k][1],  # Defines space of hyperparameters
                    # optimization algorithm, hyperotp will select its
                    # parameters automatically (Search algorithm: Tree of Parzen Estimators, a Bayesian method)
                    algo=tpe.suggest,
                    max_evals=n_iter,  # maximum number of iterations
                    trials=trials  # logging
                )

                print("\nFit {}...".format(models[k][0]))
                results = mod_eval.fit_model_h(
                    params=best_Parameters,
                    estimator=k,
                    X_train=X_train_scaled,
                    y_train=y_train,
                    X_test=X_test_scaled
                )
                predictions = results['y_pred']
                model = results['model']
                time_elapsed1 = results['time_train']
                time_elapsed2 = results['time_predict']

                areaUnderROC = mod_eval.model_evaluation_auc(
                    y_test, predictions)

                optResult = trials.results

                print("################# Iteration {} ###################".format(i + 1))
                print("Best params {}".format(best_Parameters))
                print("Hyperparameter Optimization results: {}".format(optResult))
                print(
                    "{}: Time-Elapsed Fit - '{}'".format(models[k][0], time_elapsed1))
                print(
                    "{}: Time-Elapsed Transform - '{}'".format(models[k][0], time_elapsed2))
                print(
                    "{}: Area under ROC - '{}'".format(models[k][0], areaUnderROC))
                list_auc.append(areaUnderROC)
                list_time_train.append(time_elapsed1)
                list_time_predict.append(time_elapsed2)

                dict_auc[k] = dict_auc[k] + list_auc
                dict_time_train[k] = dict_time_train[k] + list_time_train
                dict_time_predict[k] = dict_time_train[k] + list_time_predict

    print("\n----------------- Median Results -----------------")
    print("Median RF -> {}".format(statistics.median(dict_auc['RF'])))
    print("Median DT -> {}".format(statistics.median(dict_auc['DT'])))
    print("Median LR -> {}".format(statistics.median(dict_auc['LR'])))
    # print("Median GBT -> {}".format(statistics.median(dict_auc['GBT'])))
    # print("Median SVC -> {}".format(statistics.median(dict_auc['SVC'])))
    # print("Median MLP -> {}".format(statistics.median(dict_auc['MLP'])))

    results = {'AUC': dict_auc, 'Time_Train': dict_time_train,
               'Time_Predict': dict_time_predict}

    return results


def newest_main() -> None:
    
    """

    Main function

    """

    args = d_ingest.parse_command_line()
    print(args)
    pd.set_option('display.max_rows', None)
    # ags[0] -> Machine deviation file;
    if d_ingest.file_exists(args[0]) is True:

        # Read machine deviation data
        df = d_ingest.read_file_data(args[0])

        df = data_preparation(df)

        df = feature_engineering(df)

        # summarize class distribution
        counter = Counter(df['Block'])
        print(counter)

        ml_results = modeling(df)


# python app.py -f static/files/dataset.csv
if __name__ == "__main__":
    newest_main()
