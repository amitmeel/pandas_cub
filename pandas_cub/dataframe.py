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

        # Allow for special methods for strings
        self.str = StringMethods(self)
        self._add_docs()


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

    def __len__(self):
        """
        Make the builtin len function work with our dataframe

        Returns
        -------
        int: the number of rows in the dataframe
        """
        return len(next(iter(self._data.values())))

    @property
    def columns(self):
        """
        _data holds column names mapped to arrays
        take advantage of internal ordering of dictionaries to
        put columns in correct order in list. Only works in 3.6+

        Returns
        -------
        list of column names
        """
        return list(self._data)

    @columns.setter
    def columns(self, columns):
        """
        Set the column names of the dataframe.

        Parameters
        ----------
        columns: list of strings
            The new column names

        Returns
        -------
        Nones
        """
        if not isinstance(columns, list):
            raise TypeError("`columns` must be a list of strings")
        if not all(isinstance(column, str) for column in columns):
            raise TypeError("All column names must be a string")
        if len(columns) != len(self._data):
            raise ValueError("`columns` must have the same length as the number of columns in the dataframe")
        if len(columns) != len(set(columns)):
            raise ValueError("`columns` must not contain duplicate column names")
        
        new_data = dict(zip(columns, self._data.values()))
        self._data = new_data

    @property
    def shape(self):
        """
        Return the shape of the dataframe

        Returns
        -------
        tuple
            two-item tuple of number of rows and columns
        """
        return (len(self), len(self._data))

    def _repr_html_(self):
        """
        Used to create a string of HTML to nicely display the DataFrame
        in a Jupyter Notebook. Different string formatting is used for
        different data types.

        The structure of the HTML is as follows:
        <table>
            <thead>
                <tr>
                    <th>data</th>
                    ...
                    <th>data</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>{i}</strong></td>
                    <td>data</td>
                    ...
                    <td>data</td>
                </tr>
                ...
                <tr>
                    <td><strong>{i}</strong></td>
                    <td>data</td>
                    ...
                    <td>data</td>
                </tr>
            </tbody>
        </table>
        """
        html = '<table><thead><tr><th></th>'
        for col in self.columns:
            html += f"<th>{col:10}</th>"

        html += '</tr></thead>'
        html += "<tbody>"

        only_head = False
        num_head = 10
        num_tail = 10
        if len(self) <= 20:
            only_head = True
            num_head = len(self)

        for i in range(num_head):
            html += f'<tr><td><strong>{i}</strong></td>'
            for col, values in self._data.items():
                kind = values.dtype.kind
                if kind == 'f':
                    html += f'<td>{values[i]:10.3f}</td>'
                elif kind == 'b':
                    html += f'<td>{values[i]}</td>'
                elif kind == 'O':
                    v = values[i]
                    if v is None:
                        v = 'None'
                    html += f'<td>{v:10}</td>'
                else:
                    html += f'<td>{values[i]:10}</td>'
            html += '</tr>'

        if not only_head:
            html += '<tr><strong><td>...</td></strong>'
            for i in range(len(self.columns)):
                html += '<td>...</td>'
            html += '</tr>'
            for i in range(-num_tail, 0):
                html += f'<tr><td><strong>{len(self) + i}</strong></td>'
                for col, values in self._data.items():
                    kind = values.dtype.kind
                    if kind == 'f':
                        html += f'<td>{values[i]:10.3f}</td>'
                    elif kind == 'b':
                        html += f'<td>{values[i]}</td>'
                    elif kind == 'O':
                        v = values[i]
                        if v is None:
                            v = 'None'
                        html += f'<td>{v:10}</td>'
                    else:
                        html += f'<td>{values[i]:10}</td>'
                html += '</tr>'

        html += '</tbody></table>'
        return html

    @property
    def values(self):
        """
        Returns
        -------
        A single 2D NumPy array of the underlying data
        """
        return np.column_stack(self._data.values())

    @property
    def dtypes(self):
        """
        Returns
        -------
        A two-column DataFrame of column names in one column and
        their data type in the other
        """
        DTYPE_NAME = {'O': 'string', 'i': 'int', 'f': 'float', 'b': 'bool'}
        col_arr = np.array(self.columns)
        dtypes = []
        for values in self._data.values():
            kind = values.dtype.kind
            dtype = DTYPE_NAME[kind]
            dtypes.append(dtype)

        return DataFrame({'Column Name': col_arr, 'Data Type': np.array(dtypes)})

    def __getitem__(self, item):
        """
        Use the brackets operator to simultaneously select rows and columns

        A single string selects one column -> df['colname']
        A list of strings selects multiple columns -> df[['colname1', 'colname2']]
        A one column DataFrame of booleans that filters rows -> df[df_bool]

        Row and column selection simultaneously -> df[rs, cs]
            where cs and rs can be integers, slices, or a list of integers
            rs can also be a one-column boolean DataFrame

        Returns
        -------
        A subset of the original DataFrame
        """
        # select a single column -> df['colname']
        if isinstance(item, str):
            return DataFrame({item: self._data[item]})

        # select multiple columns -> df[['colname1', 'colname2']]
        if isinstance(item, list):
            return DataFrame({col: self._data[col] for col in item})

        # boolean selection
        if isinstance(item, DataFrame):
            if item.shape[1] != 1:
                raise ValueError('Can only pass a one column DataFrame for selection')

            bool_arr = next(iter(item._data.values()))
            if bool_arr.dtype.kind != 'b':
                raise TypeError('DataFrame must be a boolean')

            new_data = {}
            for col, values in self._data.items():
                new_data[col] = values[bool_arr]
            return DataFrame(new_data)

        if isinstance(item, tuple):
            return self._getitem_tuple(item)
        else:
            raise TypeError('Select with either a string, a list, or a row and column '
                            'simultaneous selection')

    def _getitem_tuple(self, item):
        # simultaneous selection of rows and cols -> df[rs, cs]
        if len(item) != 2:
            raise ValueError('Pass either a single string or a two-item tuple inside the '
                                'selection operator.')
        row_selection = item[0]
        col_selection = item[1]
        if isinstance(row_selection, int):
            row_selection = [row_selection]
        elif isinstance(row_selection, DataFrame):
            if row_selection.shape[1] != 1:
                raise ValueError('Can only pass a one column DataFrame for selection')
            row_selection = next(iter(row_selection._data.values()))
            if row_selection.dtype.kind != 'b':
                raise TypeError('DataFrame must be a boolean')
        elif not isinstance(row_selection, (list, slice)):
            raise TypeError('Row selection must be either an int, slice, list, or DataFrame')

        if isinstance(col_selection, int):
            col_selection = [self.columns[col_selection]]
        elif isinstance(col_selection, str):
            col_selection = [col_selection]
        elif isinstance(col_selection, list):
            new_col_selction = []
            for col in col_selection:
                if isinstance(col, int):
                    new_col_selction.append(self.columns[col])
                else:
                    new_col_selction.append(col)
            col_selection = new_col_selction
        elif isinstance(col_selection, slice):
            start = col_selection.start
            stop = col_selection.stop
            step = col_selection.step
            if isinstance(start, str):
                start = self.columns.index(col_selection.start)
            if isinstance(stop, str):
                stop = self.columns.index(col_selection.stop) + 1

            col_selection = self.columns[start:stop:step]
        else:
            raise TypeError('Column selection must be either an int, string, list, or slice')

        new_data = {}
        for col in col_selection:
            new_data[col] = self._data[col][row_selection]
        return DataFrame(new_data)

    def _ipython_key_completions_(self):
        # allows for tab completion when doing df['c
        return self.columns

    def __setitem__(self, key, value):
        # adds a new column or a overwrites an old column
        if not isinstance(key, str):
            raise NotImplementedError('Only able to set a single column')

        if isinstance(value, np.ndarray):
            if value.ndim != 1:
                raise ValueError('Setting array must be 1D')
            if len(value) != len(self):
                raise ValueError('Setting array must be same length as DataFrame')
        elif isinstance(value, DataFrame):
            if value.shape[1] != 1:
                raise ValueError('Setting DataFrame must be one column')
            if len(value) != len(self):
                raise ValueError('Setting and Calling DataFrames must be the same length')
            value = next(iter(value._data.values()))
        elif isinstance(value, (int, str, float, bool)):
            value = np.repeat(value, len(self))
        else:
            raise TypeError('Setting value must either be a numpy array, '
                            'DataFrame, integer, string, float, or boolean')

        if value.dtype.kind == 'U':
            value = value.astype('O')

        self._data[key] = value

    def _add_docs(self):
        agg_names = ['min', 'max', 'mean', 'median', 'sum', 'var',
                     'std', 'any', 'all', 'argmax', 'argmin']
        agg_doc = \
        """
        Find the {} of each column

        Returns
        -------
        DataFrame
        """
        for name in agg_names:
            getattr(DataFrame, name).__doc__ = agg_doc.format(name)

class StringMethods:

    def __init__(self, df):
        self._df = df

    def capitalize(self, col):
        return self._str_method(str.capitalize, col)

    def center(self, col, width, fillchar=None):
        if fillchar is None:
            fillchar = ' '
        return self._str_method(str.center, col, width, fillchar)

    def count(self, col, sub, start=None, stop=None):
        return self._str_method(str.count, col, sub, start, stop)

    def endswith(self, col, suffix, start=None, stop=None):
        return self._str_method(str.endswith, col, suffix, start, stop)

    def startswith(self, col, suffix, start=None, stop=None):
        return self._str_method(str.startswith, col, suffix, start, stop)

    def find(self, col, sub, start=None, stop=None):
        return self._str_method(str.find, col, sub, start, stop)

    def len(self, col):
        return self._str_method(str.__len__, col)

    def get(self, col, item):
        return self._str_method(str.__getitem__, col, item)

    def index(self, col, sub, start=None, stop=None):
        return self._str_method(str.index, col, sub, start, stop)

    def isalnum(self, col):
        return self._str_method(str.isalnum, col)

    def isalpha(self, col):
        return self._str_method(str.isalpha, col)

    def isdecimal(self, col):
        return self._str_method(str.isdecimal, col)

    def islower(self, col):
        return self._str_method(str.islower, col)

    def isnumeric(self, col):
        return self._str_method(str.isnumeric, col)

    def isspace(self, col):
        return self._str_method(str.isspace, col)

    def istitle(self, col):
        return self._str_method(str.istitle, col)

    def isupper(self, col):
        return self._str_method(str.isupper, col)

    def lstrip(self, col, chars):
        return self._str_method(str.lstrip, col, chars)

    def rstrip(self, col, chars):
        return self._str_method(str.rstrip, col, chars)

    def strip(self, col, chars):
        return self._str_method(str.strip, col, chars)

    def replace(self, col, old, new, count=None):
        if count is None:
            count = -1
        return self._str_method(str.replace, col, old, new, count)

    def swapcase(self, col):
        return self._str_method(str.swapcase, col)

    def title(self, col):
        return self._str_method(str.title, col)

    def lower(self, col):
        return self._str_method(str.lower, col)

    def upper(self, col):
        return self._str_method(str.upper, col)

    def zfill(self, col, width):
        return self._str_method(str.zfill, col, width)

    def encode(self, col, encoding='utf-8', errors='strict'):
        return self._str_method(str.encode, col, encoding, errors)

    def _str_method(self, method, col, *args):
        old_values = self._df._data[col]
        if old_values.dtype.kind != 'O':
            raise TypeError('The `str` accessor only works with string columns')
        new_values = []
        for val in old_values:
            if val is None:
                new_values.append(val)
            else:
                new_val = method(val, *args)
                new_values.append(new_val)
        arr = np.array(new_values)
        return DataFrame({col: arr})