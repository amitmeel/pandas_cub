import numpy as np

class DataFrame:

    def __init__(self, data):
        """
        A DataFrame holds two dimensional heterogeneous data. Create it by
        passing a dictionary of NumPy arrays to the values parameter

        Parameters
        ----------
        data: dict
            A dictionary of strings mapped to NumPy arrays. The key will
            become the column name.
        """
        
        # check for correct input types
        self._check_input_types(data)

    def _check_input_types(self, data):
        """
        Check that the input data is of the correct type.
        """
        if not isinstance(data, dict):
            raise TypeError("`data` must be a dictionary of1-D NumPy arrays")

        for column_name, values in data.items():
            if not isinstance(column_name, str):
                raise TypeError("All column names must be a string")
            if not isinstance(values, np.ndarray):
                raise TypeError("All values must be a 1-D NumPy array")
            else:
                if values.ndim != 1:
                    raise ValueError("All values must be 1-D Numpy array")