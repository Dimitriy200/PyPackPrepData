# import sys
# sys.path.append('C:/Users/Dmitriy/Desktop/Univer/Diplom/PyPackPrepData/src/')

import pandas as pd
import pytest
import sys
import os
import numpy as np

from numpy import genfromtxt
from prepData import PrepData

base_dir = os.path.abspath("diplom_autoencoder")

# # Small_BAD_test_FD001 = pd.read_csv("C:/Users/Dmitriy/Desktop/Univer/Diplom/diplom_autoencoder/data/raw/Small_BAD_test_FD001.csv")
# # inp_test_raw_json = os.path.join(base_dir, "data", "raw","2024-06-02_2024-06-03_2024-06-04") 
# # out_test_processed_json = os.path.join(base_dir, "data", "processed")

inp_data = os.path.join(base_dir, "data", "processed")
out_data = os.path.join(base_dir, "data", "final")


data = genfromtxt(os.path.join(inp_data, "ANOMALY_train_FD001.csv"), delimiter=',')


pr = PrepData()
pr.different_anomaly(data, out_data)















#________________________________________________________________________________________________________

# # def test_prepData_check_type(data: pd.DataFrame):
# #     p = PrepData()
# #     assert p.check_type(dataset = data) == False

# # test_prepData_check_type(Small_BAD_test_FD001)




# # def check_load_json():
# #     prep_data_res = PrepData.jsons_to_csv(inp_test_raw_json,
# #                                        out_test_processed_json,
# #                                        "New_csv_all_jsons.csv")
# #     print(prep_data_res)


# # check_load_json()


# def different_train_and_valid(inp_path: str,
#                                   out_path: str,
#                                   procent_train: str = 30):

#         # res_arr = []

#         file_Name_Train_DF = "Train.csv"
#         file_Name_Valid_DF = "Valid.csv"

#         list_units = os.listdir(inp_path)

#         # count_units = []

#         str_df, col_df  = genfromtxt(os.path.join(inp_path, list_units[0]), delimiter = ',').shape

#         np_train = np.zeros(col_df)
#         np_valid = np.zeros(col_df)
        

#         for unit in list_units:
            
#             unit_np_DF = genfromtxt(os.path.join(inp_path, unit), delimiter = ',')
#             # unit_numbers = unit_np_DF[:, 1]
#             unit_numbers = unit_np_DF[:, 0].tolist()
            
#             last_valid = check_min_repeate_units(unit_numbers)
            
#             last_time_cycles = array_of_outer_row_formation(unit_np_DF)
            
#             dict_val_train = different_arrays(unit_np_DF,
#                                               np_train,
#                                               np_valid,
#                                               last_valid,
#                                               last_time_cycles)
            
        
#         np.savetxt(os.path.join(out_path, file_Name_Train_DF), dict_val_train['valid_np_train'], delimiter=",")
#         np.savetxt(os.path.join(out_path, file_Name_Valid_DF), dict_val_train['valid_np_valid'], delimiter=",")

#         return os.path.join(out_path, file_Name_Train_DF)


# #Поиск минимального количества повторений в UnitNumber
# def check_min_repeate_units(unit_numbers: list,
#                             procent_train: float = 0.3):

#     last_valid = 0
#     for num in set(unit_numbers):
#         if last_valid == 0:
#             last_valid = unit_numbers.count(num)
#         elif unit_numbers.count(num) < last_valid:
#             last_valid = unit_numbers.count(num)

#     print(last_valid)
#     last_valid = last_valid * procent_train
#     last_valid = round(last_valid, 0)
    
#     return last_valid



# # Формирование массива крайних строк UnitNumber
# def array_of_outer_row_formation(unit_np_DF: np.array):

#     count_time_cycles = 1
#     last_time_cycles = []
#     count_unit_number = unit_np_DF[1, 0]
#     current_str_num = 0
#     str_df, col_df  = unit_np_DF.shape

#     for str in unit_np_DF:
#         unit_number = unit_np_DF[current_str_num, 0]
#         #print(unit_number)

#         if(count_unit_number != unit_number or current_str_num == str_df - 1):
#             last_time_cycles.append(count_time_cycles)
#             count_unit_number = unit_number
        
#         count_time_cycles += 1
#         current_str_num += 1
    
#     last_time_cycles[-1] += 1

#     return last_time_cycles



# #Разделение на два массива
# def different_arrays(unit_np_DF: np.array,
#                      np_train: np.array, 
#                      np_valid: np.array,
#                      last_valid: int,
#                      last_time_cycles: list):

#     count_unit_number = unit_np_DF[1, 0]
#     count_time_cycles = 1
#     current_str_num = 0
#     count = 0

#     for str in unit_np_DF:

#         unit_number = unit_np_DF[current_str_num, 0]

#         if(count_unit_number != unit_number):
#             count += 1
#             count_unit_number = unit_number

#         barrer = last_time_cycles[count] - last_valid

#         if(count_time_cycles < barrer):
#             np_train = np.vstack((np_train, str))
        
#         else:
#             np_valid = np.vstack((np_valid, str))

#         count_time_cycles += 1
#         current_str_num += 1

#     return {'valid_np_train': np_train,
#             'valid_np_valid': np_valid}



# print(f"\n{inp_data} \n{out_data}\n")
# different_train_and_valid(inp_data, out_data)




