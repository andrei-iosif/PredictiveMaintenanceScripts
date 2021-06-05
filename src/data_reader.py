import h5py
import numpy as np
import pandas as pd


class ColumnNames:
    def __init__(self, w_cols, x_s_cols, x_v_cols, t_cols, a_cols):
        self.w_cols = w_cols
        self.x_s_cols = x_s_cols
        self.x_v_cols = x_v_cols
        self.t_cols = t_cols
        self.a_cols = a_cols
        self.target_col = ['RUL']

    def get_all_columns_list(self):
        return self.a_cols + self.t_cols + self.x_s_cols + self.x_v_cols + self.w_cols + self.target_col


class DataReader:
    def __init__(self):
        self.train_set = None
        self.test_set = None
        self.column_names = None

    def load_dataset(self, filename, load_train=True, load_test=True):
        """
        Reads a dataset from a given .h5 file and compose (in memory) the train and test data.
        :param filename: path to the input .h5 file
        :param load_train: read and compose development dataset
        :param load_test: read and compose test dataset
        :return: void
        """
        with h5py.File(filename, 'r') as hdf:
            # Column names
            W_var = np.array(hdf.get('W_var'))
            X_s_var = np.array(hdf.get('X_s_var'))
            X_v_var = np.array(hdf.get('X_v_var'))
            T_var = np.array(hdf.get('T_var'))
            A_var = np.array(hdf.get('A_var'))

            self.column_names = ColumnNames(w_cols=list(np.array(W_var, dtype='U20')),
                                            x_s_cols=list(np.array(X_s_var, dtype='U20')),
                                            x_v_cols=list(np.array(X_v_var, dtype='U20')),
                                            t_cols=list(np.array(T_var, dtype='U20')),
                                            a_cols=list(np.array(A_var, dtype='U20')))

            column_names_list = self.column_names.get_all_columns_list()

            # Development set
            if load_train:
                W_dev = np.array(hdf.get('W_dev'))  # W
                X_s_dev = np.array(hdf.get('X_s_dev'))  # X_s
                X_v_dev = np.array(hdf.get('X_v_dev'))  # X_v
                T_dev = np.array(hdf.get('T_dev'))  # T
                Y_dev = np.array(hdf.get('Y_dev'))  # RUL
                A_dev = np.array(hdf.get('A_dev'))  # Auxiliary

                self.train_set = pd.DataFrame(data=np.concatenate((A_dev, T_dev, X_s_dev, X_v_dev, W_dev, Y_dev), axis=1),
                                              columns=column_names_list)

            # Test set
            if load_test:
                W_test = np.array(hdf.get('W_test'))  # W
                X_s_test = np.array(hdf.get('X_s_test'))  # X_s
                X_v_test = np.array(hdf.get('X_v_test'))  # X_v
                T_test = np.array(hdf.get('T_test'))  # T
                Y_test = np.array(hdf.get('Y_test'))  # RUL
                A_test = np.array(hdf.get('A_test'))  # Auxiliary

                self.test_set = pd.DataFrame(data=np.concatenate((A_test, T_test, X_s_test, X_v_test, W_test, Y_test), axis=1),
                                             columns=column_names_list)
