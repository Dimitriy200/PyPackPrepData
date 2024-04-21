import sys
sys.path.append('C:/Users/Dmitriy/Desktop/Univer/Diplom/PyPackPrepData/src/')

import pandas as pd
import pytest
import sys

from prepData import PrepData


Small_BAD_test_FD001 = pd.read_csv("C:/Users/Dmitriy/Desktop/Univer/Diplom/diplom_autoencoder/data/raw/Small_BAD_test_FD001.csv")

def test_prepData_check_type(data: pd.DataFrame):
    p = PrepData()
    assert p.check_type(dataset = data) == False

test_prepData_check_type(Small_BAD_test_FD001)