import numpy as np
from numpy.testing import assert_array_equal
import pytest

import pandas_cub as pdc

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
