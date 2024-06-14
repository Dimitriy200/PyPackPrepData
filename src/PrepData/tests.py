# import sys
# sys.path.append('C:/Users/Dmitriy/Desktop/Univer/Diplom/PyPackPrepData/src/')

import sys
import os
import numpy as np

from prepData import PrepData


base_dir = os.path.abspath("diplom_autoencoder")
dir_final = os.path.join(base_dir, "data", "final")
dir_processed = os.path.join(base_dir, "data", "processed")

# dir_normaly = os.path.join(base_dir, "data", "raw", "NORMALY")
# dir_anomaly = os.path.join(base_dir, "data", "raw", "ANOMALY")



#--------------------------------------------------------------------


def check_start_all_func_json(dir_final = dir_final,
                         dir_processed = dir_processed,
                         dir_raw = os.path.join(base_dir, "data", "raw", "2024-06-02_2024-06-03_2024-06-04")):

    pr = PrepData()
    pr.start_prepData_json(dir_raw,
                          dir_processed,
                          dir_final)

# check_start_all_func_json()

#--------------------------------------------------------------------


def check_start_all_func_csv(dir_final = dir_final,
                             dir_processed = dir_processed,
                             dir_raw = os.path.join(base_dir, "data", "raw", "for_tests")):

    pr = PrepData()
    pr.start_prepData_csv(dir_raw,
                           dir_processed,
                           dir_final)

check_start_all_func_csv()

#--------------------------------------------------------------------


def check_start_all_func_csv(dir_final = dir_final,
                         dir_processed = dir_processed,
                         dir_raw = os.path.join(base_dir, "data", "raw", "train")):
    
    pr = PrepData()

    fd1 = pr.get_np_arr_from_csv(os.path.join(dir_raw, "trainFD001.csv"))
    fd2 = pr.get_np_arr_from_csv(os.path.join(dir_raw, "trainFD002.csv"))
    fd3 = pr.get_np_arr_from_csv(os.path.join(dir_raw, "trainFD003.csv"))
    fd4 = pr.get_np_arr_from_csv(os.path.join(dir_raw, "trainFD004.csv"))

    all_fd = np.concatenate(fd1,fd2, fd3, fd4)

    


#--------------------------------------------------------------------


def check_json_to_numpy():
    
    pr = PrepData()
    # arr = pr.json_to_numpy(json_dict)
    # print(f"array = {arr}, \ntype -> {type(arr[0])}")


#--------------------------------------------------------------------


def check_dif_on_normaly(dir_normaly_fd = os.path.join(base_dir, "data", "raw", "train", "train_FD001.csv")):

    pr = PrepData()
    data = pr.get_np_arr_from_csv(dir_normaly_fd)

    pr.different_anomaly(data,
                         dir_processed)


# check_dif_on_normaly()

#--------------------------------------------------------------------

def array_of_outer_row_formation_v2(unit_np_DF: np.array):
    
    res = []
    data = unit_np_DF[:, 0]

    print(data)

    index = 0
    for n in data:
        if index == 0:
            index += 1
            continue
        
        if index == len(data) - 1:
            res.append(index)
            break
        
        if n != data[index - 1]:
            res.append(index)
        
        index += 1

    return res


#--------------------------------------------------------------------

def check_data(dir_normaly_fd = os.path.join(base_dir, "data", "raw", "train", "train_FD001.csv")):

    pr = PrepData()
    # data_csv = pr.get_np_arr_from_csv(dir_normaly_fd)
    data_json = pr.jsons_to_list(os.path.join(base_dir, "data", "raw", "2024-06-02_2024-06-03_2024-06-04"))
    data_json = np.array(data_json)
    print(data_json)

    res1 = pr.array_of_outer_row_formation(unit_np_DF=data_json)

    print(res1)

    res2 = []
    index = 0
    for n in res1:
        
        if index == 0:
            index += 1
            continue
        
        if index == len(res1) - 1:
            break
        
        res2.append(n - res1[index - 1])
        index += 1
    
    print(f"\nres = {res2}")
    print(f"min = {min(res2)}")


#--------------------------------------------------------------------

def get_last(dir_df = os.path.join(base_dir, "data", "raw", "train", "train_FD001.csv")):
    pr = PrepData()
    data = pr.get_np_arr_from_csv(dir_df)

    lasts = pr.array_of_outer_row_formation(data)
    lasts = lasts[1:]

    last_unit_values = []

    index = 0
    index_lasts = 0
    for st in data:
        
        if index == lasts[index_lasts] - 1:
            index_lasts+=1
            last_unit_values.append(st)
        
        index+=1

    last_unit_values = np.array(last_unit_values)
    np.savetxt(os.path.join(base_dir, "data", "raw", "last_values.csv"), last_unit_values, delimiter=",")

    print(f"last_unit_values = {last_unit_values}")


def get_first(dir_df = os.path.join(base_dir, "data", "raw", "train", "train_FD001.csv")):
    pr = PrepData()
    data = pr.get_np_arr_from_csv(dir_df)

    lasts = pr.array_of_outer_row_formation(data)

    last_unit_values = []

    index = 0
    index_lasts = 0
    for st in data:
        
        if index_lasts == len(lasts) - 1:
            break

        if index == lasts[index_lasts]:
            index_lasts+=1
            last_unit_values.append(st)
        
        index+=1

    last_unit_values = np.array(last_unit_values)
    np.savetxt(os.path.join(base_dir, "data", "raw", "first_values.csv"), last_unit_values, delimiter=",")


# get_first()


#--------------------------------------------------------------------

def cocate_data():

    pr = PrepData()

    fd_1 = pr.get_np_arr_from_csv(os.path.join(base_dir, "data", "raw", "train", "train_FD001.csv"))
    fd_2 = pr.get_np_arr_from_csv(os.path.join(base_dir, "data", "raw", "train", "train_FD002.csv"))
    fd_3 = pr.get_np_arr_from_csv(os.path.join(base_dir, "data", "raw", "train", "train_FD003.csv"))
    fd_4 = pr.get_np_arr_from_csv(os.path.join(base_dir, "data", "raw", "train", "train_FD004.csv"))

    res = np.concatenate(fd_1, fd_2, fd_3, fd_4)

    np.savetxt(dir_df = os.path.join(base_dir, "data", "processed", "All_FD.csv"))

# cocate_data()


#--------------------------------------------------------------------

def check_different_train_and_valid():
    pr = PrepData()
    pr.different_train_and_valid(os.path.join(base_dir, "data", "processed"),
                                 os.path.join(base_dir, "data", "final"))

# check_different_train_and_valid()

#--------------------------------------------------------------------