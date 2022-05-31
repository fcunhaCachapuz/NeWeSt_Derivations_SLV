import argparse  # https://docs.python.org/3/library/argparse.html
from os import path  # https://docs.python.org/3/library/os.path.html
import pandas as pd


def parse_command_line(
) -> str:

    """

    Function to parser from command-line options

    Returns
    -------
    :return: The file location.
    :rtype: str

    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-file_path",
        metavar='-f',
        type=str,
        nargs="+",
        help="directory from where read data regarding weighings and machine deviation",
    )
    parser_args = parser.parse_args()

    return parser_args.file_path


def file_exists(
    location: str
) -> bool:

    """

    Function to verify is file exists

    Parameters
    ----------
    :param location: The file location.
    :type location: str

    Returns
    -------
    :return: True is file exists and otherwise False.
    :rtype: bool

    """

    # Verify if is not a file according to the given file path
    if not path.isfile(location):
        return False

    return True


def read_file_data(
    location: str
) -> pd.DataFrame:

    """
    
    Function of read a all file lines to a list. Each index of a list corresponds to one line

    Parameters
    ----------
    :param location: The file location.
    :type location: str

    Returns
    -------
    :return: Pandas Dataframe of file contents read.
    :rtype: pd.Dataframe

    """

    file_data = pd.DataFrame
    try:
        file_data = pd.read_csv(r'{}'.format(location),
                                on_bad_lines='skip', sep=';', encoding='latin-1')
    except IOError:
        print("An Exception occurred when reading the file!!")

    return file_data
