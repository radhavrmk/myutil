import pandas as pd
from os import path
import pickle
import matplotlib.pyplot as plt

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s')
# logging.basicConfig(filename='./app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


def check_file_header(fnames, nlines=5):
    """
    This function prints the first few lines from each of the files passed as argument. This
    function helps to quickly inspect the file before loading through Pandas
    or some other library

    Arguments:
    fnames: list of file names with complete path or list of Path objects
    nlines: number of lines to print

    """
    from itertools import islice
    for fname in fnames:
        print(f"\nPrinting header from {fname} \n#########################################")
        with open(fname) as f:
            head = list(islice(f, nlines))
        for line in head:
            print(line)


def display_all(df):
    """
    Utility to override the Pandas defaults temporarily and display all rows and columns
    :param df: dataframe to be displayed
    :return:
    """
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        display(df)


def save_object(obj, filename):
    """
    Utility to save a python object to file using pickle. Not suitable for very large objects
    Overwrites existing the file already exists
    :param obj: Object to be saved
    :param filename: file name
    :return:
    """
    with open(filename, 'wb') as output_file:  # Overwrites any existing file.
        pickle.dump(obj, output_file, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    """
    Load a python object stored in a file using pickle
    :param filename: File from which to load the object
    :return:
    """
    with open(filename, 'rb') as input_file:  # Overwrites any existing file.
        obj = pickle.load(input_file)
    return obj

# Test case for save and load utilities above
# config_dictionary = {'remote_hostname': 'google.com', 'remote_port': 80}
# save_object(config_dictionary, path.join(CONFIG_PATH, "test.cfg"))
# my_dict = load_object(path.join(CONFIG_PATH, "test.cfg"))
# assert(my_dict == config_dictionary)


class ConfigSaver():
    """
    Utility class to store configuration values and objects as a dictionary
    The object persists itself upon creation and on addition/updates
    """
    def __init__(self, filename):
        #         super().__init__()
        self.filename = filename
        self.config = {}

        if path.exists(filename):
            self.config = load_object(filename)
        else:
            new_path = path.dirname(filename)
            if not path.exists(new_path):
                os.makedirs(new_path)
            save_object(self.config, filename)

    def get(self, key):
        return self.config.get(key)

    def get_keys(self):
        return self.config.keys()

    def update(self, key, value):
        self.config[key] = value
        save_object(self.config, self.filename)

    def delete(self, key):
        self.config.pop(key, None)
        save_object(self.config, self.filename)

# Test case for ConfigSaver class above
# file_name = "./data/config/ConfigSaverTest.cfg"
# cfg = ConfigSaver(file_name)
# cfg.update("Test", [1,2,3,4,5])
# cfg.update("Test2", [1,2,3,4,5])
# assert(cfg.get("Test") == [1,2,3,4,5])
# cfg.update("Test", [1,2,3,4])
# assert(cfg.get("Test") == [1,2,3,4,])
# cfg.delete("Test")
# assert(cfg.get("Test") is None)


def get_columns(df, data_type="category"):
    """
    Utility to get columns of a specific data type from dataframe
    :param df: dataframe contaning the data
    :param data_type: data type of columns that need to be returned
    :return: list of columns that are of specified data type
    """
    if data_type == "numeric":
        cols = [col_name for col_name, col_type in df.dtypes.items()  if col_type.kind in ["i", "f"]]
    elif data_type == "integer":
        cols = [col_name for col_name, col_type in df.dtypes.items()  if col_type.kind == "i"]
    elif data_type == "float":
        cols = [col_name for col_name, col_type in df.dtypes.items()  if col_type.kind == "f"]
    elif data_type in ["object", "category"] :
        cols = df.columns[df.dtypes == data_type].values
    elif data_type == "non_numeric":
        cols = [col_name for col_name, col_type in df.dtypes.items()  if col_type.kind == "O"]
    elif data_type == "date":
        cols = [col_name for col_name, col_type in df.dtypes.items()  if col_type.kind == "M"]
    return cols


def convert_to_cats(df, columns=None, dtype="object", nunique=6, na_string=None,
                    lookup=False, file="./convert_to_cats.cfg"):
    """
    This method uses Pandas Categorical to convert a column to Categorical. Alternatively this can
    be changed to use LabelEncoder or some other method
    Note that Pandas will encode NaN as -1

    The category encoding for each column is saved to file and can be used on validation set or
    during inference time

    Arguments:
    :param df: Dataframe
    :param columns: columns to be converted. If none all columns considered
    :param dtype: data type of columns to be considered for conversion if list of columns not given
    as input
    :param nunique: Max number of unique values in a column to be included for conversion
    :param na_string: Value to be used to fill NA values. replace with np.nan if
    need to retain nan values
    :param lookup: Used for validation and test sets. Whether to use the lookup values
    and use from prior conversion
    :param file: File to store/retrieve the category mappings

    """
    converted_cols = []

    if nunique == -1:
        nunique = len(df)

    if not columns:
        columns = get_columns(df, dtype)

    cat_dict = ConfigSaver(file)

    if lookup:
        pass

    for col_name in columns:
        if len(df[col_name].unique()) <= nunique:
            if lookup:
                categories = cat_dict.get("convert_to_cats_" + col_name)
                na_string = cat_dict.get("convert_to_cats_" + col_name, "NA_String")
            else:
                categories = list(df[col_name].dropna().unique())
                if na_string:
                    categories = [na_string] + categories
                cat_dict.update("convert_to_cats_" + col_name, categories)
                cat_dict.update("convert_to_cats_" + col_name + "NA_String", na_string)

            if na_string:
                df[col_name] = df[col_name].fillna(na_string)
            df[col_name] = pd.Categorical(df[col_name], ordered=True)
            converted_cols.append(col_name)

    logging.info(f"columns converted to Categorical type :\n {converted_cols}")
    return df


def get_low_unique_value_columns(df, cutoff=5, exclude_categorical=True, display_stats=True):
    """
    Utility to quickly look at columns that have low number of unique values. These are potential
    candidates to convert to categorical type. By default the method excludes columns that are
    already categorical
    :param df: data in the form of Pandas dataframe
    :param cutoff: cutoff value. Number of unique values should be less than or equal to this cutoff,
    for a column to be selected
    :param exclude_categorical: Whether to exclude columns that are already categorical, defaults to True
    :return col_list: list of columns
    """
    df_length = len(df)

    if exclude_categorical:
        columns = [(col_name, col_type) for col_name, col_type in df.dtypes.items() if col_type.name is not "category"]

    ##** Note: rewrite using dataframe and add a plot

    col_list = []

    for col_name, col_type in columns:
        unique_count = df[col_name].nunique()

        if (unique_count <= cutoff):
            values = (col_name, col_type, unique_count)
            if display_stats:
                print(f"\nColumn: {values[0]}  data type: {values[1]}  No of unique values: {values[2]}" )
                print(df[col_name].unique())
            col_list.append(col_name)

    return col_list


def impute_numerical(df, strategy="median", add_na_column = True, threshold_pct=5):
    numeric_cols = get_columns(df, "numeric")
    cols_to_impute = []
    for col in numeric_cols:
        na_count = df[col].isnull().sum()
        pct_na = na_count/len(df[col])

        if pct_na > threshold_pct:
            df[col+"_na"] = df[col].isnull()

            df[col] = df[col].fillna(df[col].median())
            print(df[col+"_na"].value_counts())

    return df


def peek_object_columns(df):
    cat_cols = get_columns(df, "object")
    for col in cat_cols:
        print(f"\n********** {(col, len(df[col].unique()))} **************")
        print(df[col].unique()[:20])


def encode_categorical(df):
    """
    Compresses the size of the dataframe by changing column data type to minimum required size
    that can accommodate the values. This is done only for categorical columns.
    If a numeric column is expected to have very few values, change that to categorical first
    and then use this method.
    :param df: input dataframe
    :return: modified dataframe
    """
    cat_cols = df.select_dtypes("category").columns
    for col in cat_cols:
        df[col] = df[col].cat.codes + 1
        unique_no = len(df[col].unique())
        if unique_no < 50:
            df[col] = df[col].astype("uint8")
        elif unique_no < 16000:
            df[col] = df[col].astype("int16")
        else:
            df[col] = df[col].astype("int32")
    return df


# missing value counter
def na_counter(df, display_plot=True, display_as_list=True):
    """
    Utility to display the count and percent of NA values by column in a dataframe
    :param df: input dataframe
    :param display_plot: whether to display the plot
    :param display_as_list: whether to list dataframe after the picture
    :return: returns a dataframe containing counts and percent values
    """
    coln = list(df.columns)
    columns = ['NA Count', "PERCENT"]

    # if df.isnull().any().sum() == 0:
    #     return pd.DataFrame(columns=columns)

    na_dict = {}
    for col in coln:
        if df[col].isnull().any():
            na_count = df[col].isnull().sum()
            na_percent = df[col].isnull().sum() / len(df)
            na_dict[col] = [na_count, na_percent]

    if len(na_dict) == 0:
        return pd.DataFrame()

    df_na = pd.DataFrame(na_dict).T
    df_na.columns = columns
    df_na.index.name = "Column"

    df_na = df_na.sort_values(by="PERCENT", ascending=True)

    print(f"No of columns in dataframe: {len(df.columns)}")
    print(f"No of columns with NA values: {len(df_na)}")

    if display_plot:
        if df_na.shape[0] < 30:
            plt.rcParams['figure.figsize'] = (16.0, 8.0)
            df_na.PERCENT.plot(kind="bar", title="% na values")
        else:
            plt.rcParams['figure.figsize'] = (8,30)
            df_na.PERCENT.plot(kind="barh",title="% na values")
        plt.show()

    if display_as_list is True:
        display_all(df_na)

    return df_na


def get_id_columns(df):
    """
    Utility to identify columns that are some type of IDs and have unique values in every row
    These columns can be deleted as they add no useful information for ML model
    :param df: dataframe to check for ID columns
    :return: list of ID columns
    """
    id_cols = []
    cols= df.columns
    df_len = len(df)
    for column in cols:
        if df[column].nunique() == df_len:
            id_cols.append(column)

    return id_cols


def drop_id_columns(df):
    """
    Utility to drop ID columns - that are some type of IDs and have unique values in every row
    These columns can be deleted as they add no useful information for ML model
    :param df: dataframe to check for ID columns
    :return: dataframe with ID columns deleted
    """
    id_cols = get_id_columns(df)
    if len(id_cols) > 0:
        df = df.drop(id_cols, axis = 1)

    return df

def find_outlier_range(dfs: pd.Series, iqr_factor=1.5, clip_low=None, clip_high=None):
    """
    Function to get a range from a Pandas series that is considered non-outlier. Values beyond
    this range can be considered outliers

    Arguments :
    dfs : Pandas series
    iqr_factor : factor by which 25% and 75% quantiles multiplied to get the outlier range
    clip_low : lower bound at which values to be clipped. Useful when values are bounded, eg: Prices
    clip_high : upper bound at which values to be clipped. Useful when values are bounded, eg: Age

    """
    quant1 = dfs.quantile(0.25)
    quant3 = dfs.quantile(0.75)
    IQR = quant3 - quant1
    low = quant1 - iqr_factor * IQR
    high = quant3 + iqr_factor * IQR

    if clip_low is not None:
        low = max(clip_low, low)
    if clip_high is not None:
        high = min(clip_high, high)
    return (low, high)

# # Quick test case for above function
# xx = pd.Series(np.random.randint(1,500, 40))
# find_outlier_range(xx, clip_low=0, clip_high=500)


def rmse(y, y_pred):
    Z = y - y_pred
    return -np.sqrt(np.sum(Z**2)/len(Z))

def mse(y, y_pred):
    Z = y - y_pred
    return -(np.sum(Z**2)/len(Z))

def mad(y, y_pred):
    Z = y - y_pred
    return -pd.Series(Z).mad()


def print_scores(model, x_train, y_train, x_valid, y_valid):
    pass


def plot_image(image):
    """
    Utility to draw image using matplotlib
    :param image: Image to be displayed
    :return:
    """
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")


def plot_color_image(image):
    """
    Utility to draw image using matplotlib
    :param image: Image to be displayed
    :return:
    """
    plt.imshow(image, interpolation="nearest")
    plt.axis("off")

    
def get_top_and_bottom_features(feature_importances, feature_names, count=20):
    """
    Utility to display feature importances as bar charts
    :param model: trained model object
    :param count: number of highest and least important features to display
    :return: return the highest and least important features as a tuple of pandas series
    """

    coef_pd = pd.Series(feature_importances, index = feature_names).sort_values(ascending=False)
    imp_coef = coef_pd.head(count)
    least_coef = coef_pd.abs().tail(count)
    
    plt.rcParams['figure.figsize'] = (16.0, 8.0)

    plt.subplot(1,2,1)
    plt.tight_layout(pad=0.4,w_pad=0.5, h_pad=1.0)
    plt.title("Leastx important Coefficients")
    least_coef.plot(kind = "barh")

    plt.subplot(1,2,2)
    plt.tight_layout(pad=0.4,w_pad=0.5, h_pad=1.0)
    plt.title("Most important Coefficients")
    imp_coef.plot(kind = "barh")

    plt.show()
    return (imp_coef, least_coef)

# def plot_series(time, series, format="-", start=0, end=None, label=None):
#     """
#     Utility to draw time series chart
#     :param time: The time axis data
#     :param series: Series values
#     :param format: format to be used by matplotlib plot() method
#     :param start: Starting point
#     :param end: ending point
#     :param label: Title for the chart
#     :return:
#     """
#     """
#
#     :param image: Image to be displayed
#     :return:
#     """
#     plt.plot(time[start:end], series[start:end], format, label=label)
#     plt.xlabel("Time")
#     plt.ylabel("Value")
#     if label:
#         plt.legend(fontsize=14)
#     plt.grid(True)