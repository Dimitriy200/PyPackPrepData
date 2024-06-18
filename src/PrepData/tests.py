# import sys
# sys.path.append('C:/Users/Dmitriy/Desktop/Univer/Diplom/PyPackPrepData/src/')

import sys
import os
import numpy as np
import logging

from prepData import PrepData

logging.basicConfig(level=logging.INFO,
                    filename=os.path.join(os.path.abspath("preprocess-data"),"src", "prepData", "logs", "prep_data_logs.log" ),
                    filemode="w",
                    format="%(asctime)s %(levelname)s %(message)s")

base_dir = os.path.abspath("diplom_autoencoder")
dir_final = os.path.join(base_dir, "data", "final")
dir_processed = os.path.join(base_dir, "data", "processed")
dir_raw = os.path.join(base_dir, "data", "raw")

# dir_normaly = os.path.join(base_dir, "data", "raw", "NORMALY")
# dir_anomaly = os.path.join(base_dir, "data", "raw", "ANOMALY")



#--------------------------------------------------------------------
def repl_numpy():
    pr = PrepData()
    arr = np.array(
        [[1, 2, 2],
         [1, 3, 4],
         [2, 3, 4],
         [3, 7, 5]])
    
    # a = arr[:, 0] + 1
    # b = arr[:, 1:]

    # res = np.column_stack((a, b))

    res = arr[1:, :]

    print(res)

# repl_numpy()

#--------------------------------------------------------------------

def diff_arr():
    arr = [5, 10, 14, 20, 35, 40] # res = [5, ]
    res_arr = []

    clk = 0
    for i in arr:
        if i == arr[0]:
            res_arr.append(i)
            clk += 1
        else:
            buf = arr[clk - 1]
            res_arr.append(i - buf)
            clk += 1
    
    print(res_arr)

# diff_arr()

#--------------------------------------------------------------------

def del_arr():
    pr = PrepData()
    arr = np.array(
        [[1, 2, 2],
         [np.nan, np.nan, np.nan],
         [2, 3, 4],
         [3, 7, 5]])
    
    res_arr = np.isnan(arr[:, :])
    print(res_arr)

# del_arr()

#--------------------------------------------------------------------


def repl_df():
    pr = PrepData()
    # FD_001 - 100
    # FD_002 - 260
    # FD_003 - 100
    # FD_004 - 249

    fd_1 = pr.get_np_arr_from_csv(os.path.join(dir_raw, "train", "train_FD001.csv"))
    fd_2 = pr.get_np_arr_from_csv(os.path.join(dir_raw, "train", "train_FD002.csv"))
    fd_3 = pr.get_np_arr_from_csv(os.path.join(dir_raw, "train", "train_FD003.csv"))
    fd_4 = pr.get_np_arr_from_csv(os.path.join(dir_raw, "train", "train_FD004.csv"))

    
    un_2 = fd_2[:, 0] + 100
    buf_2 = fd_2[:, 1:]
    res_fd_2 = np.column_stack((un_2, buf_2))

    un_3 = fd_3[:, 0] + 100 + 260
    buf_3 = fd_3[:, 1:]
    res_fd_3 = np.column_stack((un_3, buf_3))

    un_4 = fd_4[:, 0] + 100 + 260 + 100
    buf_4 = fd_4[:, 1:]
    res_fd_4 = np.column_stack((un_4, buf_4))

    print(f"un_2 = {res_fd_2[1, 0]}")
    print(f"un_3 = {res_fd_3[1, 0]}")
    print(f"un_4 = {res_fd_4[1, 0]}")

    # res = np.concatenate([fd_1, res_fd_2, res_fd_3, res_fd_4])

    # np.savetxt(os.path.join(dir_raw, "new_train", "new_all_train.csv"), res, delimiter=',')


    # np.savetxt(os.path.join(dir_raw, "new_train", "new_train_FD001.csv"), fd_1, delimiter=',')
    np.savetxt(os.path.join(dir_raw, "new_train", "new_train_FD002.csv"), res_fd_2, delimiter=',')
    np.savetxt(os.path.join(dir_raw, "new_train", "new_train_FD003.csv"), res_fd_3, delimiter=',')
    np.savetxt(os.path.join(dir_raw, "new_train", "new_train_FD004.csv"), res_fd_4,  delimiter=',')

# repl_df()

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

file_train = os.path.join(base_dir, "data", "raw", "new_train", "new_all_train.csv")
file_test = os.path.join(base_dir, "data", "raw", "new_train", "new_train_FD001.csv")

def check_start_prepData_csv(dir_final = dir_final,
                             dir_processed = dir_processed,
                             dir_raw = file_test): #prep_new_all_train.csv
    
    pr = PrepData()
    pr.start_prepData_csv(path_raw = dir_raw,
                          path_processed = dir_processed,
                          path_final = dir_final,
                          last_procent = 0.3)

check_start_prepData_csv()

#--------------------------------------------------------------------



def check_different_anomaly(dir_final = dir_final,
                             dir_processed = dir_processed,
                             dir_raw = os.path.join(base_dir, "data", "raw",),
                             last_procent = 0.2):

    pr = PrepData()
    data = pr.get_np_arr_from_csv(os.path.join(dir_raw, "new_train", "new_train_FD001.csv")) #new_train_FD001.csv

    out_path = os.path.join(dir_raw, "tests")
    
    d_norm_and_anom = pr.different_anomaly(dataFrame=data,
                                            out_path_normal = out_path,
                                            out_path_anomal = out_path,
                                            Name_Normal_DF = "test_norm.csv",
                                            Name_Anomal_DF = "test_anom.csv",
                                            last_procent = 0.7)
    

    logging.info(f"d_norm_and_anom [Normal] = {d_norm_and_anom['Normal'].shape}")
    logging.info(f"d_norm_and_anom [Anomal] = {d_norm_and_anom['Anomal'].shape}")
    np.savetxt(os.path.join(out_path, "test_Norm_1.csv"), d_norm_and_anom['Normal'], delimiter=",")
    np.savetxt(os.path.join(out_path, "test_Anom_1.csv"), d_norm_and_anom['Anomal'], delimiter=",")


    
    np_anom = d_norm_and_anom['Anomal']
    d_new_norm_and_anom = pr.different_anomaly(dataFrame = np_anom,
                                            out_path_normal = out_path,
                                            out_path_anomal = out_path,
                                            Name_Normal_DF = "test_norm.csv",
                                            Name_Anomal_DF = "test_anom.csv",
                                            last_procent = 0.1)
    

    logging.info(f"d_new_norm_and_anom [Normal] = {d_new_norm_and_anom['Normal'].shape}")
    logging.info(f"d_new_norm_and_anom [Anomal] = {d_new_norm_and_anom['Anomal'].shape}")
    np.savetxt(os.path.join(out_path, "test_Norm_2.csv"), d_new_norm_and_anom['Normal'], delimiter=",")
    np.savetxt(os.path.join(out_path, "test_Anom_2.csv"), d_new_norm_and_anom['Anomal'], delimiter=",")

    

# check_different_anomaly()

#--------------------------------------------------------------------


def check_employ_pipeline():

    pr = PrepData()
    data = pr.get_np_arr_from_csv(os.path.join(dir_raw, "new_train", "new_all_train.csv"))

    prep_data = pr.employ_Pipline(array_np = data)

    np.savetxt(os.path.join(dir_raw, "new_train", "prep_new_all_train.csv"), prep_data, delimiter=',')


# check_employ_pipeline()
#--------------------------------------------------------------------


def check_start_prepData_for_add_traine(dir_final = os.path.join(dir_final, "train_test_json"),
                                        dir_raw = os.path.join(base_dir, "data", "raw", "2024-06-02_2024-06-03_2024-06-04")):

    pr = PrepData()
    pr.start_prepData_for_add_traine(path_raw = dir_raw,
                                     path_final = dir_final,
                                     Name_file_train = "train_json.csv",
                                     Name_file_test = "test_json.csv")

# check_start_prepData_for_add_traine()

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