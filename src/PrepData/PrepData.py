import os
import pickle
import numpy as np
import json
import csv
import sys
import logging

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from numpy import genfromtxt
from sklearn.model_selection import train_test_split

from typing import Dict, List, Any



class PrepData:


    def __init__(self):
        #11
        self.__defaultPipline = self.createDefault_Pipline()
        
        # Код состояния для функций
        self.status: bool

    # Вызов дефолтного Pipline - внутренняя функция
    @property
    def defaultPipline(self):
        return self.__defaultPipline
    

    @classmethod
    def start_prepData_json(self,
                            path_raw: str,
                            path_processed: str,
                            path_final: str):
        

        # Объединяем json-ы в единый датасет
        print("START READ DATA FROM JSON")
        dataset_csv = self.jsons_to_list(path_raw)


        # Преобразуем датасет в numpy array
        dataset_np = np.array(dataset_csv)

        self.process_different_data(dataset_np = dataset_np,
                                    path_raw = path_raw,
                                    path_processed = path_processed,
                                    path_final = path_final)


    @classmethod
    def start_prepData_csv(self,
                           path_raw: str,
                           path_processed: str,
                           path_final: str):
        

        # считываем даннные из всех csv
        print("START READ DATA FROM CSV")
        list_csv = os.listdir(path_raw)
        if len(list_csv) > 0:
            all_np_csv = genfromtxt(os.path.join(path_raw, list_csv[0]), delimiter=',')
            
            for csv in list_csv:
                if csv == list_csv[0]:
                    continue
                
                buffer_np = genfromtxt(os.path.join(path_raw, csv), delimiter=',')
                all_np_csv = np.concatenate((all_np_csv, buffer_np), axis=0)


        self.process_different_data(dataset_np = all_np_csv,
                                    path_raw = path_raw,
                                    path_processed = path_processed,
                                    path_final = path_final)


    @classmethod
    def process_different_data(self,
                               dataset_np: np.array,
                               path_raw: str,
                               path_processed: str,
                               path_final: str):
        

        # Получаем стандартизированный и нормированный датасет
        print("START PIPELINE")
        prep_dataset = self.employ_Pipline(dataset_np)


        # Разделяем датасет на нормальные и аномальные значеиня
        print("START DIFFERENT TO NORMAL AND ANOMAL")

        name_dir_norm_data = "normal"
        name_dir_anom_data = "anomal"

        name_file_norm_data = "Normal.csv"
        name_file_anom_data = "Anomal.csv"

        path_normal = os.path.join(path_processed, name_dir_norm_data)
        path_anomal = os.path.join(path_processed, name_dir_anom_data)

        self.different_anomaly(prep_dataset,
                               out_path_normal = path_normal,
                               out_path_anomal = path_anomal,
                               Name_Normal_DF = name_file_norm_data,
                               Name_Anomal_DF = name_file_anom_data)   
        

        # Делим ANOM данные в соотношении 80/20, где 20% - данные для статичесской валидации
        print("START DIFFERENT ANOMAL TO 80 / 20 (STATIC VALIDATE)")

        path_static_data = os.path.join(path_final, "static_valid")

        name_new_Anomal_data_file = "New_Anomal.csv"
        name_static_valid_Anomal_data_file = "Satic_validation_Anomal.csv"
        
        first_diff_Anomal_data, static_valid_Anomal_data = self.different_data(diff_file_csv_path = os.path.join(path_processed, name_dir_anom_data, name_file_anom_data),
                                                                               new_file_name_1 = name_new_Anomal_data_file,
                                                                               new_file_name_2 = name_static_valid_Anomal_data_file,
                                                                               out_path_file_1 = "",
                                                                               out_path_file_2 = path_static_data)
        

        # Делим NORM данные в соотношении 80/20, где 20% - данные для статичесской валидации
        print("START DIFFERENT NORMAL TO 80 / 20 (STATIC VALIDATE)")

        name_new_Normal_data_file = "Choice_barrier_normal.csv"
        name_static_valid_Normal_data_file = "Control_barrier_normal.csv"

        first_diff_Normal_data, static_valid_Normal_data = self.different_data(diff_file_csv_path = os.path.join(path_processed, name_dir_anom_data, name_file_anom_data),
                                                             new_file_name_1 = name_new_Normal_data_file,
                                                             new_file_name_2 = name_static_valid_Normal_data_file,
                                                             out_path_file_1 = "",
                                                             out_path_file_2 = path_static_data)


        # Делим оставшиеся ANOM данные на данные для подбора и контроля барьера
        print("START DIFFERENT ANOMAL TO 80 / 20 (CHOISE AND CONTROL BARRIER)")
        
        path_barrier_data = os.path.join(path_processed, "search_barrier")

        name_file_anomal_choise_barrier = "Choise_barrier_Anomal.csv"
        name_file_anomal_control_barrier = "Control_barrier_Anomal.csv"

        ch_barrier_anomal, cntr_barrier_anomal = self.different_data(inp_data = first_diff_Anomal_data,
                                                                     new_file_name_1 = name_file_anomal_choise_barrier,
                                                                     new_file_name_2 = name_file_anomal_control_barrier,
                                                                     out_path_file_1 = path_barrier_data,
                                                                     out_path_file_2 = path_barrier_data)


        # Вычисляем Процент для деления NORM данных для их разделения в следующем шаге
        print("START COUNT PROCENT ANORMAL FROM NORMAL")

        str_first_diff_Anomal_data, col_first_diff_Anomal_data  = first_diff_Anomal_data.shape
        str_first_diff_Normal_data, col_first_diff_Normal_data = first_diff_Normal_data.shape

        procent_Anom_from_Norm_barrier = str_first_diff_Anomal_data / str_first_diff_Normal_data


        # Отделяеем от NORM часть, равную Anom_barriers
        print("START DIFFERENT NORMAL TO 80 / 20 (CHOISE AND CONTROL BARRIER)")
        
        sec_diff_Normal_data, df_for_Norm_barrier = self.different_data(inp_data = first_diff_Normal_data,
                                                                        procent_train = procent_Anom_from_Norm_barrier)


        # Делим оставшиеся NORM данные на данные для подбора и контроля барьера
        print("START DIFFERENT NORMAL TO 80 / 20 (CHOISE AND CONTROL BARRIER)")

        name_file_Normal_choise_barrier = "Choise_barrier_Normal.csv"
        name_file_Normal_control_barrier = "Control_barrier_Normal.csv"

        ch_barrier_Normal, cntr_barrier_Normal = self.different_data(inp_data = df_for_Norm_barrier,
                                                                     new_file_name_1 = name_file_Normal_choise_barrier,
                                                                     new_file_name_2 = name_file_Normal_control_barrier,
                                                                     out_path_file_1 = path_barrier_data,
                                                                     out_path_file_2 = path_barrier_data)
        
        # Делим Оставшиеся NORM ДАННЫЕ НА TRAIN и TEST
        print("START DIFFERENT NORMAL TO 80 / 20 (TRAIN AND TEST)")

        name_traine_Norm_data = "train.csv"
        name_test_Norm_data = "test.csv"
        path_train_test_data = os.path.join(path_final, "train_and_test")

        norm_train, norm_test = self.different_data(inp_data = sec_diff_Normal_data,
                                                    new_file_name_1 = name_traine_Norm_data,
                                                    new_file_name_2 = name_test_Norm_data,
                                                    out_path_file_1 = path_train_test_data,
                                                    out_path_file_2 = path_train_test_data)


    #@staticmethod
    @classmethod
    def createDefault_Pipline(self) -> Pipeline:
        
        status_log = ["Create pipline successfull", "Create pipline error"]

        try:
            simple_inputer = KNNImputer(n_neighbors = 2)
            std_scaler = StandardScaler()
            pipe_num = Pipeline([('imputer', simple_inputer), ('scaler', std_scaler)])

            print(status_log[0])
            self.status = True
            return pipe_num
        
        except:
            print(status_log[1])
            self.status = False
            return False
    

    @classmethod
    def save_Pipeline(self, saved_pipeline: Pipeline,
                      save_path: str) -> bool:
        status_log = ["Save pipline successfull", "Save pipline error"]

        try:
            with open(save_path, 'wb') as handle:
                save_pik_pipeline = pickle.dumps(saved_pipeline)
            
            self.out_info(True, status_log[0])
            return self.status
        
        except:
            self.out_info(False, status_log[1])
            return self.status


    @classmethod
    def load_Pipeline(self, load_path: str) -> Pipeline:

        status_log = ["Load pipline successfull", "Load pipline error"]

        try:
            with open(load_path, 'rb') as handle:
                save_pik_pipeline = pickle.load(handle)

            self.out_info(True, status_log[0])
            return save_pik_pipeline
        
        except:
            self.out_info(False, status_log[1])
            return 0


    @classmethod
    def fit_pipeline(self,
                     pipeline: Pipeline,
                     fit_data: np.array)-> Pipeline:
        
        status_log = ["Check data and fit pipeline successfull", "Check data and fit pipeline error"]

        try:
            while (self.is_nan_dataFrame_Line(fit_data) == False):
                fit_data = self.to_standardization_df(fit_data)
                    
            pipeline = self.start_fit_pipeline(pipeline, fit_data)

            self.out_info(True, status_log[0])
            return pipeline
        
        except:
            self.out_info(False, status_log[1])
            return self.status


    @classmethod
    def employ_Pipline(self,
                        # inp_dir: str,       # B inpFilesList и outFilesList указывать полный путь
                        array_np: np.array,
                        #out_dir: str,
                        # new_csv_file_name: str,
                        pipline: Pipeline = defaultPipline,) -> np.array:
        
        status_log  =           ["Preprocess data finished successfull",        "Preprocess data finished error"]
        get_doc_log =           ["Getting a list of documents...",              "Documents have been received"]

        # На конце выходной строки дирректории должна стоять "/"
        # if inp_dir[-1] != '/':
        #     inp_dir = f"{inp_dir}/"
        # if out_dir[-1] != '/':
        #     out_dir = f"{out_dir}/"

        # if ".csv" not in new_csv_file_name:
        #     new_csv_file_name = f"{new_csv_file_name}.csv"

        # Получаем все документы в папке
        # print(get_doc_log[0])
        # inpFilesList = os.listdir(inp_dir)
        # outFilesList = inpFilesList
        
        # Вывод списка файлов в диррекории
        # for fl in inpFilesList:
        #     print("   .../" + fl)
            
        # print(get_doc_log[1])

        # Применение
        # for file in inpFilesList:
        #     if file == ".gitkeep":
        #         continue
            
        #     print("  Processed --> ", file)

        #     # Загружаем i-тый фаил
        #     dataFrame = genfromtxt(os.path.join(inp_dir, file), delimiter = ',')
        
        
        array_np = self.to_standardization_df(array_np)
            
        # Проверяем датасет на пригодность (отсутствие пропусков)
        # for line in array_np:
        #     if (self.is_nan_dataFrame_Line(line) == False):
        #         print("Dataset is BAD Starting standartization dataframe...")
        #         self.to_standardization_df(array_np)
        #     else:
        #         continue
        
        print("Dataset is GOOD  Starting employ pipeline...")
        
        # newDataFrame = pd.DataFrame(dataFrame, columns = pipline['scaler'].get_feature_names_out(dataFrame.columns))
        # print(array_np)
        # array_nd = np.array(array_np)
        
        new_pipeline = self.createDefault_Pipline()
        newDataFrame = new_pipeline.fit_transform(array_np)
        # newDataFrame = pipline.fit_predict(array_np)

        # Сохранение
        # new_csv_dir_path = os.path.join(out_dir, new_csv_file_name)
        # np.savetxt(new_csv_dir_path, newDataFrame, delimiter = ",")

        res_dataframe = np.array(newDataFrame)
        
        return res_dataframe


    @classmethod
    def different_anomaly(self,
                          dataFrame: np.array,
                          out_path_normal: str,
                          out_path_anomal: str,
                          Name_Normal_DF,
                          Name_Anomal_DF,
                          last_procent: int = 0.2) -> dict:

        # list_units = os.listdir(inp_path)


        str_df, col_df = dataFrame.shape

        np_train = np.zeros(col_df)
        np_valid = np.zeros(col_df)

        # for unit in list_units:

            # unit_numbers = unit_np_DF[:, 1]
        unit_numbers = dataFrame[:, 0].tolist()

        last_time_cycles = self.array_of_outer_row_formation(dataFrame)

        last_valid = self.check_min_repeate_units(last_time_cycles,
                                                    procent_quitting = last_procent)
        
        dict_val_train = self.different_arrays(dataFrame,
                                               np_train,
                                               np_valid,
                                               last_valid,
                                               last_time_cycles)
        
        np.savetxt(os.path.join(out_path_normal, Name_Normal_DF), dict_val_train['Normal'], delimiter=",")
        np.savetxt(os.path.join(out_path_anomal, Name_Anomal_DF), dict_val_train['Anomal'], delimiter=",")

        return


    @classmethod
    def different_train_and_valid(self,
                                  inp_path: str,
                                  out_path: str,
                                  procent_train: float = .7) -> str:

        Name_Train_Normal_DF =  "Train_Normal.csv"
        Name_Test_Normal_DF  =  "Test_Normal.csv"
        Name_Valid_Anomal_DF =  "Valid_Anomal.csv"

        list_units = os.listdir(inp_path)
    
        for file in list_units:  
            
            if file == "Normal.csv":
                norm_data_np = genfromtxt(os.path.join(inp_path, file), delimiter = ',')

                if not np.any(norm_data_np):
                    print(f"Массив Normal.csv пустой")
                    continue
                else:
                    train, test = train_test_split(norm_data_np,
                                                random_state=0,
                                                train_size = procent_train)
                    
                    np.savetxt(os.path.join(out_path, Name_Train_Normal_DF), train, delimiter=",")
                    np.savetxt(os.path.join(out_path, Name_Test_Normal_DF), test, delimiter=",")
            
            elif file == "Anomal.csv":
                anom_data_np = genfromtxt(os.path.join(inp_path, file), delimiter = ',')

                if not np.any(anom_data_np):
                    print(f"Массив Anomal.csv пустой")
                    continue
                else:
                    np.savetxt(os.path.join(out_path, Name_Valid_Anomal_DF),
                            anom_data_np, delimiter=",")

        return os.path.join(out_path, Name_Train_Normal_DF)


    @classmethod
    def different_data(self,
                       new_file_name_1: str,
                       new_file_name_2: str,
                       out_path_file_1: str,
                       out_path_file_2: str,
                       diff_file_csv_path: str = "",
                       inp_data: np.array = None,
                       procent_train: float = .8):

        if (diff_file_csv_path == "") and (inp_data != None):
            diff_df = inp_data
        
        elif (diff_file_csv_path != "") and (inp_data == None):
            diff_df = genfromtxt(diff_file_csv_path, delimiter=',')
        
        else:
            print("ОШИБКА, ВЫБЕРИТЕ ЛИБО ЗАГРУЗКУ ИЗ ФАЙЛА, ЛИБО ИЗ ОБЪЕКТА")
            return [0], [0]

        df_1, df_2 = train_test_split(diff_df,
                                      random_state=0,
                                      train_size = procent_train)
        
        if out_path_file_1 != "":
            np.savetxt(os.path.join(out_path_file_1, new_file_name_1),
                       df_1, delimiter=",")
        
        if out_path_file_2 != "":
            np.savetxt(os.path.join(out_path_file_2, new_file_name_2),
                       df_2, delimiter=",")
        
        return df_1, df_2


    @classmethod
    def jsons_to_list(self, inp_json_dir: str) -> list:

        res_arr = []

        # Получить список подпапок с json-ами
        list_units = os.listdir(inp_json_dir)
        list_units.sort()

        logging.info(f"List of units: {list_units}")
        # print(list_units)

        # Пробежаться по json-ам и склеить их
        # print(jsons)
        
        for unit in list_units:
            unit_dir = os.path.join(inp_json_dir, unit)
            logging.info(f"Unit dir: {unit_dir}")
            # print(unit_dir)
            
            jsons_files = os.listdir(unit_dir)
            jsons_files.sort()

            logging.info(f"Json files in unit: {unit_dir}")
            # print(jsons_files)

            for file in jsons_files:
                json_dict = {}
                jsons_dir = os.path.join(unit_dir, file)
                
                with open(jsons_dir, "r") as json_file:
                    json_dict = json.load(json_file)
                    value_buffer = list(json_dict.values())

                    value_buffer = value_buffer[:-3]

                    value_buffer = list(map(float, value_buffer))
                    res_arr.append(value_buffer)
        
        # Сохранить в указанную папку
        # csv_dir_path = os.path.join(out_csv_dir, name_out_csv)
        # with open(csv_dir_path, "w") as new_csv:
        #     write = csv.writer(new_csv, lineterminator='\n')
        #     write.writerows(res_arr)

        
        
        return res_arr


    @classmethod
    def concate_data(inp_csv_dir: str,
                     out_csv_dir: str,
                     name_out_csv: str,
                     col_count: int = 0):

        inpFilesList = os.listdir(inp_csv_dir)
        
        if inpFilesList == []:
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

        df = genfromtxt(os.path.join(inp_csv_dir, inpFilesList[0]))
        col, str = df.shape

        if col_count == 0:
            res = np.zeros(col)
        else:
            res = np.zeros(col_count)

        for doc in inpFilesList:
            file = genfromtxt(os.path.join(inp_csv_dir, doc), delimiter = ',')

            res = np.vstack(res, file)

        np.savetxt(os.path.join(out_csv_dir, name_out_csv),
                   res,
                   delimiter = ",")


    @classmethod
    def json_to_numpy(self, json_dict_list: List[Dict[str, Any]]):

        res_arr = []

        for one_dict in json_dict_list:
            json_dict = list(one_dict.values())
            buffer = json_dict[:-3]
            res_arr.append(list(map(float, buffer)))

        res_arr = np.array(res_arr)

        return res_arr


        

#           --- Service Functions --- 


    @classmethod
    def start_fit_pipeline(self,
                            pipeline: Pipeline, 
                            fit_data: np.array) -> Pipeline:
        
        status_log = ["Fit pipline successfull", "Fit pipline error"]

        try:
            new_pipeline = pipeline.fit(fit_data)

            self.out_info(True, status_log[0])
            return new_pipeline
        
        except:
            self.out_info(False, status_log[1])
            return 0


    @classmethod
    def to_standardization_df(self, dataFrame: np.array) -> np.array:
    
        status_log = ["Standartization dataframe successfull", "Standartization dataframe error"]

        # dataFrame = self.add_col_indexes(dataFrame)
        dataFrame = self.delete_names(dataFrame)
        dataFrame = self.delete_nan_str(dataFrame)

        self.out_info(True, status_log[0])
        return dataFrame


    # 1 - Добавление столбца с индексами
    @classmethod
    def add_col_indexes(self,
                        dataFrame: np.array) -> np.array:
        
        status_log = ["Add column with indexes in dataframe successfull", "Add column with indexes in dataframe error"]

        # try:
        str_count, col_count = dataFrame.shape
        indexses = []
        coef_if_empty = 0

        print("str -->", str_count,
            "\ncol -->", col_count)

        if (self.is_nan_dataFrame_Line(dataFrame[0, :])):
            indexses.append(np.nan)
            coef_if_empty = 1
        
        for i in range(str_count - coef_if_empty):
            indexses.append(i)

        dataFrame = np.insert(dataFrame, 0, indexses, axis= 1)

        self.out_info(True, status_log[0])
        return dataFrame
        
        # except:
        #     self.out_info(False, status_log[1])
        #     return self.status


    # 2 - Удаление 0-й строки с названиями столбцов
    @classmethod
    def delete_names(self, dataFrame: np.array) -> np.array:
        
        status_log = ["Delete columns with names in dataframe successfull", "Delete columns with names in dataframe error"]

        try:
            if(self.is_named_col(dataFrame) == True):
                dataFrame = dataFrame[1:, 0:]

            self.out_info(True, status_log[0])
            return dataFrame
        
        except:
            self.out_info(False, status_log[1])
            return self.status


    # 3 - Удаление строк с пропусками
    @classmethod
    def delete_nan_str(self, dataFrame: np.array) -> np.array:
    
        status_log = ["Delete NULL lines in dataframe successfull", "Delete NULL lines in dataframe error"]
        status = False

        str, col  = dataFrame.shape
        print(f"str = {str}\ncol = {col}")
        res_dataFrame = np.zeros(col)

        for line in dataFrame:
            if(self.is_nan_dataFrame_Line(line) == False):
                res_dataFrame = np.vstack((res_dataFrame, line))
                #print(f"res_dataFrame = {res_dataFrame}")
            else:
                continue

        res_dataFrame = res_dataFrame[1:, :]
        
        self.out_info(True, status_log[0])
        return res_dataFrame


    @classmethod
    def is_nan_dataFrame_Line(self, dataset: np.array) -> bool:

        status_log = ["Check NULL values in dataframe successfull", "Check NULL values in dataframe error"]
        self.status = False

        for st in dataset:
            if(np.isnan(st)):
                self.status = True
                #self.out_info(self.status, status_log[0] + f" result is {self.status}")
                return self.status
                
            #self.out_info(self.status, status_log[0] + f" result is {self.status}")
        return self.status


    @classmethod
    def is_named_col(self, dataset: np.array) -> bool:

        status_log = ["Check first string dataframe successfull", "Check first string dataframe error"]
        self.status = False
        first_str = dataset[0, :]

        for val in first_str:
            if(np.isnan(val)):
                self.status = True
            else:
                self.status = False
                return self.status
        
        return self.status


    @classmethod
    #Поиск минимального количества повторений в UnitNumber
    def check_min_repeate_units(self,
                                last_time_cycles: list,
                                procent_quitting: float):

        # last_valid = 0
        # for num in set(unit_numbers):
        #     if last_valid == 0:
        #         last_valid = unit_numbers.count(num)
        #     elif unit_numbers.count(num) < last_valid:
        #         last_valid = unit_numbers.count(num)

        # print(last_valid)
        # last_valid = last_valid * procent_train
        # last_valid = round(last_valid, 0)

        # res = []
        # index = 0
        # for n in unit_numbers:
            
        #     if index == 0:
        #         index += 1
        #         continue
            
        #     if index == len(unit_numbers) - 1:
        #         break
            
        #     res.append(n - unit_numbers[index - 1])
        #     index += 1
        
        # [1, ..., ...]
        last_time_cycles = last_time_cycles[1:]

        last_valid = min(last_time_cycles)

        last_valid = last_valid * procent_quitting
        last_valid = round(last_valid, 0)

        print(f"last_valid = {last_valid}")

        return last_valid


    @classmethod
    # Формирование массива крайних строк UnitNumber
    def array_of_outer_row_formation(self,
                                     unit_np_DF: np.array):

        # # count_time_cycles = 1
        # last_time_cycles = []
        # # count_unit_number = unit_np_DF[1, 0]
        # count_unit_number = 0
        # # current_str_num = 0
        # # str_df, col_df  = unit_np_DF.shape

        # for index, str in enumerate(unit_np_DF):
        #     unit_number = unit_np_DF[index, 0]
        #     #print(unit_number)
        #     if index == len(unit_np_DF) - 1:
        #         last_time_cycles.append(index)
        #         continue

        #     if count_unit_number == 0:
        #         count_unit_number = unit_number
        #         continue

        #     # if(count_unit_number != unit_number or current_str_num == str_df - 1):
        #     if(count_unit_number != unit_number):
        #         last_time_cycles.append(index - 1)
        #         count_unit_number = unit_number
            
        #     # count_time_cycles += 1
        #     # current_str_num += 1
        
        # # last_time_cycles[-1] += 1

        # return last_time_cycles


        res = []
        data = unit_np_DF[:, 0]

        print(data)

        index = 0
        for n in data:
            if index == 0:
                index += 1
                continue
            
            if index == len(data)-1:
                res.append(index + 1)
                break
            
            if n != data[index - 1]:
                res.append(index)
            
            index += 1

        print(f"outer_row = {res}")

        return res


#1 - 0    [[1, 1, 1]
#2 - 1     [1, 1, 1]
#3 - 2     [2, 2, 2]
#4 - 3     [2, 2, 2]
#5 - 4     [2, 2, 2]
#6 - 5     [2, 2, 2]
#7 - 6     [3, 3, 3]     
#8 - 7     [3, 3, 3]     unit_number
#9 - 8     [3, 3, 3]
#10 - 9    [3, 3, 3]     barrer
#11 - 10   [4, 4, 4]     count_unit_number
#12 - 11   [4, 4, 4]
#13 - 12   [4, 4, 4]
#14 - 13   [4, 4, 4]
#15 - 14   [5, 5, 5]
#16 - 15   [5, 5, 5]]

# last_time_cycles = [3, 7, 11, 17]




    @classmethod
    #Разделение на два массива
    def different_arrays(self,
                         unit_np_DF: np.array,
                         np_train: np.array, 
                         np_valid: np.array,
                         last_valid: int,
                         last_time_cycles: list):

        # count_unit_number = unit_np_DF[1, 0]
        count_unit_number = 0
        count_time_cycles = 1
        current_str_num = 0
        count = 0

        for index, str in enumerate(unit_np_DF):

            unit_number = unit_np_DF[index, 0]

            # if index == len(unit_np_DF) - 1:
                
            #     continue
            
            if count_unit_number == 0:
                count_unit_number = unit_number
                continue

            if count_unit_number != unit_number:
                if index == len(unit_np_DF) - 1:
                    np_train = np.vstack((np_train, str))
                    continue
                count += 1
                count_unit_number = unit_number

            barrer = last_time_cycles[count] - last_valid

            if index <= barrer:
                np_train = np.vstack((np_train, str))
            
            else:
                np_valid = np.vstack((np_valid, str))

            # count_time_cycles += 1
            # current_str_num += 1

        return {'Mean': np_train,
                'Small': np_valid}


    @classmethod
    def out_info(self, status: bool, text: str):
        print(text)
        self.status = status


    @classmethod
    def get_np_arr_from_csv(self, path_cfv: str) -> np.array:
        res = genfromtxt(path_cfv, delimiter=',')
        return res



#_______________________________________________________________________________________________

    #               Norm                                                   Anom
    #           80      Val_model 20                                   80      Val_model 20
# Подбор_Барьера x шт    Контроль_барьера x' шт      Подбор_Барьера x = 80(80)      Контроль барьера x' = 20(80)
    #       80 -(x + x')
    # Train 80     Test 20


    # Подбор_Барьера (объединить)
    # Контроль_барьера (объединить) для обоих ввести метки

    # алгоритм 
    # 1. делить датасет на норм и аном данные   [v]
    # 2. делить и норм и аном на 2 датасета в соотношениии 20/80. 20е - откладываем для статич валидации.    [v]
    # 3. из 80% Аномальных делить на Подбор_Барьера x = 80(80) и Контроль барьера x' = 20(80). Разметить.    [v] []
    # 4. из 80% норм формируем x и x' шт данных. [v]
    # 5. формируем два минидатасета на подбор барьера и контроль барьера. каждый содержит одинаковое кол-во норм и аном данных в x+x и x'+x' сооств. [v]
    # 5.1 на датасете train 80 обучаем модель автокодир-к
    # 5.2 валидируем на датасете test 20
    # 6. пропускаем датасет подбор баръера через модель.
    # 7. получаем массив ошибок mse
    # 8. цикл для каждого положения разделяющей поверхности от mse min до mse max делать:
    #       { для кадого объекта определяем считать его нормальным или аномальным в соответствии с положением разд поверхности
    #         для каждого объекта определяем: к какому классу ошибок он относится. TP FP TN FN.
#             расчитываем метрику для датасета.
#             запоминаем знач метрики соответсв данному положению разд поверхн}

    # 9. Определяем оптимальное положение разд поверхн-ти соответств макс знач метрики либо соотв середине интервала с макс знач метрик
    # 10. формируем модель классификации(бин). каждый объект пропуск через автокодир-к получаем mse и определяем меньше он или больше оптимального отношения разд пов-ти.
    # 11. Пропускаем все объекты датасета контроль барьера через модель классификации и определяем итоговую метрику.


    # Пример. Положит класс - аном - 1
    # подбор барьера:
    # [x1(norm), x2(norm), x3(anom), x4(anom)] -> model -> mse:[10, 20, 40, 30]
    # Разд поверхн на данной итер цикла = 35
    # модель определила: (x1 - norm, x2- norm, x3 - anom, x4 - norm)
    # x1: TN
    # x2: TN
    # x3: TP
    # x4: FN
    # metrics f1 = 0,9 или auc = ;
    # 
    # [x1(norm), x2(norm), x3(anom), x4(anom)] -> model -> mse:[10, 20, 40, 30]
    # Разд поверхн на данной итер цикла = 45
    # модель определила: (x1 - norm, x2- norm, x3 - norm, x4 - norm)
    # x1: TN
    # x2: TN
    # x3: FN
    # x4: FN
    # metrics f1 = 0,8 или auc = ;