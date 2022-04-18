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
