import numpy as np
from numpy.testing import assert_array_equal
import pytest

import pandas_cub as pdc
from ..tests import assert_df_equals

pytestmark = pytest.mark.filterwarnings("ignore")

import pandas_cub as pdc

a = np.array(['a', 'b', 'c'])
b = np.array(['c', 'd', None])
c = np.random.rand(3)
d = np.array([True, False, True])
e = np.array([1, 2, 3])
df = pdc.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})
class TestDataFrameCreation:
    
    def test_input_types(self):
        with pytest.raises(TypeError):
            pdc.DataFrame([1, 2, 3])

        with pytest.raises(TypeError):
            pdc.DataFrame({1: 5, 'b': 10})

        with pytest.raises(TypeError):
            pdc.DataFrame({'a': np.array([1]), 'b': 10})

        with pytest.raises(ValueError):
            pdc.DataFrame({'a': np.array([1]), 
                           'b': np.array([[1]])})

        # correct construction. no error
        pdc.DataFrame({'a': np.array([1]), 
                       'b': np.array([1])})

    def test_array_length(self):
        with pytest.raises(ValueError):
            pdc.DataFrame({'a': np.array([1, 2]), 
                           'b': np.array([1])})
        # correct construction. no error                           
        pdc.DataFrame({'a': np.array([1, 2]), 
                        'b': np.array([5, 10])})

    def test_convert_unicode_to_object(self):
        a_object = a.astype('O')
        assert_array_equal(df._data['a'], a_object)
        assert_array_equal(df._data['b'], b)
        assert_array_equal(df._data['c'], c)
        assert_array_equal(df._data['d'], d)
        assert_array_equal(df._data['e'], e)

    def test_len(self):
        assert len(df) == 3

    def test_columns(self):
        assert df.columns == ['a', 'b', 'c', 'd', 'e']

    def test_set_columns(self):
        with pytest.raises(TypeError):
            df.columns = 5

        with pytest.raises(ValueError):
            df.columns = ['a', 'b']

        with pytest.raises(TypeError):
            df.columns = [1, 2, 3, 4, 5]

        with pytest.raises(ValueError):
            df.columns = ['f', 'f', 'g', 'h', 'i']

        df.columns = ['f', 'g', 'h', 'i', 'j']
        assert df.columns == ['f', 'g', 'h', 'i', 'j']

        # set it back
        df.columns = ['a', 'b', 'c', 'd', 'e']
        assert df.columns == ['a', 'b', 'c', 'd', 'e']

    def test_shape(self):
        assert df.shape == (3, 5)

    def test_values(self):
        values = np.column_stack((a, b, c, d, e))
        assert_array_equal(df.values, values)

    def test_dtypes(self):
        cols = np.array(['a', 'b', 'c', 'd', 'e'], dtype='O')
        dtypes = np.array(['string', 'string', 'float', 'bool', 'int'], dtype='O')

        df_result = df.dtypes
        df_answer = pdc.DataFrame({'Column Name': cols,
                                   'Data Type': dtypes})
        assert_df_equals(df_result, df_answer)


class TestSelection:

    def test_one_column(self):
        assert_array_equal(df['a'].values[:, 0], a)
        assert_array_equal(df['c'].values[:, 0], c)

    def test_multiple_columns(self):
        cols = ['a', 'c']
        df_result = df[cols]
        df_answer = pdc.DataFrame({'a': a, 'c': c})
        assert_df_equals(df_result, df_answer)

    def test_simple_boolean(self):
        bool_arr = np.array([True, False, False])
        df_bool = pdc.DataFrame({'col': bool_arr})
        df_result = df[df_bool]
        df_answer = pdc.DataFrame({'a': a[bool_arr], 'b': b[bool_arr], 
                                   'c': c[bool_arr], 'd': d[bool_arr], 
                                   'e': e[bool_arr]})
        assert_df_equals(df_result, df_answer)

        with pytest.raises(ValueError):
            df_bool = pdc.DataFrame({'col': bool_arr, 'col2': bool_arr})
            df[df_bool]

        with pytest.raises(TypeError):
            df_bool = pdc.DataFrame({'col': np.array[1, 2, 3]})

    def test_one_column_tuple(self):
        assert_df_equals(df[:, 'a'], pdc.DataFrame({'a': a}))

    def test_multiple_columns_tuple(self):
        cols = ['a', 'c']
        df_result = df[:, cols]
        df_answer = pdc.DataFrame({'a': a, 'c': c})
        assert_df_equals(df_result, df_answer)

    def test_int_selcetion(self):
        assert_df_equals(df[:, 3], pdc.DataFrame({'d': d}))

    def test_simultaneous_tuple(self):
        with pytest.raises(TypeError):
            s = set()
            df[s]

        with pytest.raises(ValueError):
            df[1, 2, 3]

    def test_single_element(self):
        df_answer = pdc.DataFrame({'e': np.array([2])})
        assert_df_equals(df[1, 'e'], df_answer)

    def test_all_row_selections(self):
        df1 = pdc.DataFrame({'a': np.array([True, False, True]),
                             'b': np.array([1, 3, 5])})
        with pytest.raises(ValueError):
            df[df1, 'e']

        with pytest.raises(TypeError):
            df[df1['b'], 'c']

        df_result = df[df1['a'], 'c']
        df_answer = pdc.DataFrame({'c': c[[True, False, True]]})
        assert_df_equals(df_result, df_answer)

        df_result = df[[1, 2], 0]
        df_answer = pdc.DataFrame({'a': a[[1, 2]]})
        assert_df_equals(df_result, df_answer)

        df_result = df[1:, 0]
        assert_df_equals(df_result, df_answer)

    def test_list_columns(self):
        df_answer = pdc.DataFrame({'c': c, 'e': e})
        assert_df_equals(df[:, [2, 4]], df_answer)
        assert_df_equals(df[:, [2, 'e']], df_answer)
        assert_df_equals(df[:, ['c', 'e']], df_answer)

        df_result = df[2, ['a', 'e']]
        df_answer = pdc.DataFrame({'a': a[[2]], 'e': e[[2]]})
        assert_df_equals(df_result, df_answer)

        df_answer = pdc.DataFrame({'c': c[[1, 2]], 'e': e[[1, 2]]})
        assert_df_equals(df[[1, 2], ['c', 'e']], df_answer)

        df1 = pdc.DataFrame({'a': np.array([True, False, True]),
                             'b': np.array([1, 3, 5])})
        df_answer = pdc.DataFrame({'c': c[[0, 2]], 'e': e[[0, 2]]})
        assert_df_equals(df[df1['a'], ['c', 'e']], df_answer)

    def test_col_slice(self):
        df_answer = pdc.DataFrame({'a': a, 'b': b, 'c': c})
        assert_df_equals(df[:, :3], df_answer)

        df_answer = pdc.DataFrame({'a': a[::2], 'b': b[::2], 'c': c[::2]})
        assert_df_equals(df[::2, :3], df_answer)

        df_answer = pdc.DataFrame({'a': a[::2], 'b': b[::2], 'c': c[::2], 'd': d[::2], 'e': e[::2]})
        assert_df_equals(df[::2, :], df_answer)

        with pytest.raises(TypeError):
            df[:, set()]