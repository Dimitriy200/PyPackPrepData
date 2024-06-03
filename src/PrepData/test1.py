import sys
sys.path.append('C:/Users/Dmitriy/Desktop/Univer/Diplom/PyPackPrepData/src/')

import pandas as pd
import pytest
import sys
import os
import numpy as np

from prepData import PrepData

base_dir = os.path.abspath("diplom_autoencoder")

# Small_BAD_test_FD001 = pd.read_csv("C:/Users/Dmitriy/Desktop/Univer/Diplom/diplom_autoencoder/data/raw/Small_BAD_test_FD001.csv")
# inp_test_raw_json = os.path.join(base_dir, "data", "raw","2024-06-02_2024-06-03_2024-06-04") 
# out_test_processed_json = os.path.join(base_dir, "data", "processed")

inp_data = os.path.join(base_dir, "data", "processed")
out_data = os.path.join(base_dir, "data", "final")

# def test_prepData_check_type(data: pd.DataFrame):
#     p = PrepData()
#     assert p.check_type(dataset = data) == False

# test_prepData_check_type(Small_BAD_test_FD001)




# def check_load_json():
#     prep_data_res = PrepData.jsons_to_csv(inp_test_raw_json,
#                                        out_test_processed_json,
#                                        "New_csv_all_jsons.csv")
#     print(prep_data_res)


# check_load_json()


def check_different_train_and_valid():
    PrepData.different_train_and_valid(inp_data, out_data)