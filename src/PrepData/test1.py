import sys
sys.path.append('C:/Users/Dmitriy/Desktop/Univer/Diplom/PyPackPrepData/src/')

import pandas as pd
import pytest
import sys
import os
import numpy as np

from numpy import genfromtxt
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


def different_train_and_valid(inp_path: str,
                                  out_path: str,
                                  procent_train: str = 30):

        res_arr = []

        file_Name_Train_DF = "Train.csv"
        file_Name_Valid_DF = "Valid.csv"

        list_units = os.listdir(inp_path)

        last_time_cycles = []
        count_units = []

        str_df, col_df  = genfromtxt(os.path.join(inp_path, list_units[0]), delimiter = ',').shape

        valid_np_train = np.zeros(col_df)
        valid_np_valid = np.zeros(col_df)
        

        for unit in list_units:
            
            unit_np_DF = genfromtxt(os.path.join(inp_path, unit), delimiter = ',')
            
            unit_numbers = unit_np_DF[:, 1]
            str_df, col_df  = unit_np_DF.shape
            last_valid = 0
            count = 0
            current_str_num = 0
            count_time_cycles = 1
            count_unit_number = unit_np_DF[1, 0]

            unit_numbers = unit_np_DF[:, 0].tolist()

            for num in set(unit_numbers):
                if last_valid == 0:
                    last_valid = unit_numbers.count(num)
                elif unit_numbers.count(num) < last_valid:
                    last_valid = unit_numbers.count(num)

            print(last_valid)
            last_valid = last_valid / procent_train
            last_valid = round(last_valid, 0)

            # Формирование массива крайних строк UnitNumber
            for str in unit_np_DF:
                unit_number = unit_np_DF[current_str_num, 0]
                #print(unit_number)

                if(count_unit_number != unit_number or current_str_num == str_df - 1):
                    last_time_cycles.append(count_time_cycles)
                    count_unit_number = unit_number
                
                count_time_cycles += 1
                current_str_num += 1


            count_unit_number = unit_np_DF[1, 0]
            count_time_cycles = 1
            current_str_num = 0

            #Разделение 
            for str in unit_np_DF:

                unit_number = unit_np_DF[current_str_num, 0]

                if(count_unit_number != unit_number):
                    count += 1
                    count_unit_number = unit_number

                barrer = last_time_cycles[count] - last_valid

                if(count_time_cycles < barrer):
                    valid_np_train = np.vstack((valid_np_train, str))
                
                else:
                    valid_np_valid = np.vstack((valid_np_valid, str))

                count_time_cycles += 1
                current_str_num += 1

        np.savetxt(os.path.join(out_path, file_Name_Train_DF), valid_np_train, delimiter=",")
        np.savetxt(os.path.join(out_path, file_Name_Valid_DF), valid_np_valid, delimiter=",")

        return True




print(f"\n{inp_data} \n{out_data}\n")
different_train_and_valid(inp_data, out_data)