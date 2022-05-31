import pandas as pd
from typing import List


def save_dataframe_to_csv(
    df: pd.DataFrame,
    location: str
) -> None:

    """

    Function save Dataframe to csv file

    Parameters
    ----------
    :param df: The Dataframe of Weighings.
    :type df: pd.DataFrame

    :param location: The location to save the file.
    :type location: str

    Returns
    -------
    :return: None

    """

    df.to_csv(location, sep=';', index=False)


def number_input_validation(
    message: str
) -> int:

    """

    Function to validate input

    Parameters
    ----------
    :param message: The message to be displayed.
    :type message: str

    Returns
    -------
    :return: Number introduced by user.
    :rtype: int

    """

    while True:
        try:
            number = int(input(message))
        except ValueError:
            print("Select a valid option! Try again.")
            continue
        else:
            return number
            break


def show_menu(
    message: str,
    options: List
) -> int:
    """
    Function to display the menu

    Parameters
    ----------
    :param message: The message to be displayed.
    :type message: str

    Returns
    -------
    :return: Number introduced by user.
    :rtype: int

    """

    print("\n########################################")
    print("  NEWEST - {} ".format(message))
    print("########################################\n")
    for i in range(0, len(options), 1):
        print("  {} - {}".format(i, options[i]))
    print("\n########################################\n")
    selected = number_input_validation("Select a option: ")
    while selected < 0 or selected >= len(options):
        print(
            "Select a valid option [0-{}]! Try again.".format(len(options) - 1))
        selected = number_input_validation("Select a option: ")
    return selected
