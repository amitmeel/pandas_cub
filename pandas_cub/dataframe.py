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

        # check for equal array lengths
        self._check_array_lengths(data)

        # convert unicode arrays to object
        self._data = self._convert_unicode_to_object(data)

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

    def _check_array_lengths(self, data):
        """
        Check that all arrays in the input data have the same length.
        """
        lengths = [len(values) for values in data.values()]
        if len(set(lengths)) != 1:
            raise ValueError("All arrays (column values) must have the same length")

    def _convert_unicode_to_object(self, data):
        """
        Convert unicode arrays to object arrays.
        """
        for column_name, values in data.items():
            if values.dtype.kind == 'U':
                data[column_name] = values.astype(object)
        return data