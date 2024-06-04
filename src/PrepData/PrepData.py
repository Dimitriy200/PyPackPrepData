import pandas as pd
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
    def save_Pipeline(self, saved_pipeline: Pipeline, save_path: str) -> bool:
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
                        #new_csv_file_name: str,
                        pipline: Pipeline = defaultPipline,) -> np.array:
        
        status_log  =           ["Preprocess data finished successfull",        "Preprocess data finished error"]
        get_doc_log =           ["Getting a list of documents...",              "Documents have been received"]

        # На конце выходной строки дирректории должна стоять "/"
        # if inp_dir[-1] != '/':
        #     inp_dir = f"{inp_dir}/"
        # if out_dir[-1] != '/':
        #     out_dir = f"{out_dir}/"

        if ".csv" not in new_csv_file_name:
            new_csv_file_name = f"{new_csv_file_name}.csv"

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
        #     self.to_standardization_df(dataFrame)
            
            # Проверяем датасет на пригодность (отсутствие пропусков)
        for line in array_np:
            if (self.is_nan_dataFrame_Line(line) == False):
                print("Dataset is BAD Starting standartization dataframe...")
                self.is_nan_dataFrame_Line(line)
            else:
                continue
        
        print("Dataset is GOOD  Starting employ pipeline...")
        
        # newDataFrame = pd.DataFrame(dataFrame, columns = pipline['scaler'].get_feature_names_out(dataFrame.columns))
        newDataFrame = pipline.fit_transform(array_np)

        # Сохранение
        # new_csv_dir_path = os.path.join(out_dir, new_csv_file_name)
        # np.savetxt(new_csv_dir_path, newDataFrame, delimiter = ",")

        res_dataframe = np.array(newDataFrame)
        
        return res_dataframe


    @classmethod
    def different_anomaly(self,
                          dataFrame: np.array,
                          out_path: str,
                          last_procent: int = 0.1) -> bool:
    
        Name_Normal_DF = "Normal.csv"
        Name_Anomal_DF = "Anomal.csv"

        # list_units = os.listdir(inp_path)


        str_df, col_df = dataFrame.shape

        np_train = np.zeros(col_df)
        np_valid = np.zeros(col_df)
        

        # for unit in list_units:

            # unit_numbers = unit_np_DF[:, 1]
        unit_numbers = dataFrame[:, 0].tolist()
        
        last_valid = self.check_min_repeate_units(unit_numbers,
                                                    procent_train = last_procent)
        
        last_time_cycles = self.array_of_outer_row_formation(dataFrame)
        
        dict_val_train = self.different_arrays(dataFrame,
                                               np_train,
                                               np_valid,
                                               last_valid,
                                               last_time_cycles)
        
        np.savetxt(os.path.join(out_path, Name_Normal_DF), dict_val_train['valid_np_train'], delimiter=",")
        np.savetxt(os.path.join(out_path, Name_Anomal_DF), dict_val_train['valid_np_valid'], delimiter=",")

        return {"fileName_Normal_DF": Name_Normal_DF,
                "fileName_Anomal_DF": Name_Anomal_DF}


    @classmethod
    def different_train_and_valid(self,
                                  inp_path: str,
                                  out_path: str,
                                  procent_train: str = 30):

        file_Name_Train_DF = "Train.csv"
        file_Name_Valid_DF = "Valid.csv"

        list_units = os.listdir(inp_path)

        str_df, col_df  = genfromtxt(os.path.join(inp_path, list_units[0]), delimiter = ',').shape

        np_train = np.zeros(col_df)
        np_valid = np.zeros(col_df)
        

        for unit in list_units:
            
            unit_np_DF = genfromtxt(os.path.join(inp_path, unit), delimiter = ',')
            # unit_numbers = unit_np_DF[:, 1]
            unit_numbers = unit_np_DF[:, 0].tolist()
            
            last_valid = self.check_min_repeate_units(unit_numbers)
            
            last_time_cycles = self.array_of_outer_row_formation(unit_np_DF)
            
            dict_val_train = self.different_arrays(unit_np_DF,
                                              np_train,
                                              np_valid,
                                              last_valid,
                                              last_time_cycles)
            
        
        np.savetxt(os.path.join(out_path, file_Name_Train_DF), dict_val_train['valid_np_train'], delimiter=",")
        np.savetxt(os.path.join(out_path, file_Name_Valid_DF), dict_val_train['valid_np_valid'], delimiter=",")

        return os.path.join(out_path, file_Name_Train_DF)

    @classmethod
    def jsons_to_csv(self, inp_json_dir: str):

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
    def start_prepData(self,
                       path_raw: str,
                       path_processed: str,
                       path_final: str):
        
        dataset_csv = self.jsons_to_csv(path_raw)
        dataset_np = np.array(dataset_csv)

        prep_dataset = self.employ_Pipline(dataset_np)

        self.different_anomaly(prep_dataset, path_processed)
        
        self.different_train_and_valid(path_processed, path_final)


        

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

        try:
            dataFrame = self.add_col_indexes(dataFrame)
            dataFrame = self.delete_names(dataFrame)
            dataFrame = self.delete_nan_str(dataFrame)

            self.out_info(True, status_log[0])
            return dataFrame
        
        except:
            self.out_info(False, status_log[1])
            return self.status


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
                                unit_numbers: list,
                                procent_train: float = 0.3):

        last_valid = 0
        for num in set(unit_numbers):
            if last_valid == 0:
                last_valid = unit_numbers.count(num)
            elif unit_numbers.count(num) < last_valid:
                last_valid = unit_numbers.count(num)

        print(last_valid)
        last_valid = last_valid * procent_train
        last_valid = round(last_valid, 0)
        
        return last_valid


    @classmethod
    # Формирование массива крайних строк UnitNumber
    def array_of_outer_row_formation(self,
                                     unit_np_DF: np.array):

        count_time_cycles = 1
        last_time_cycles = []
        count_unit_number = unit_np_DF[1, 0]
        current_str_num = 0
        str_df, col_df  = unit_np_DF.shape

        for str in unit_np_DF:
            unit_number = unit_np_DF[current_str_num, 0]
            #print(unit_number)

            if(count_unit_number != unit_number or current_str_num == str_df - 1):
                last_time_cycles.append(count_time_cycles)
                count_unit_number = unit_number
            
            count_time_cycles += 1
            current_str_num += 1
        
        last_time_cycles[-1] += 1

        return last_time_cycles


    @classmethod
    #Разделение на два массива
    def different_arrays(self,
                         unit_np_DF: np.array,
                         np_train: np.array, 
                         np_valid: np.array,
                         last_valid: int,
                         last_time_cycles: list):

        count_unit_number = unit_np_DF[1, 0]
        count_time_cycles = 1
        current_str_num = 0
        count = 0

        for str in unit_np_DF:

            unit_number = unit_np_DF[current_str_num, 0]

            if(count_unit_number != unit_number):
                count += 1
                count_unit_number = unit_number

            barrer = last_time_cycles[count] - last_valid

            if(count_time_cycles < barrer):
                np_train = np.vstack((np_train, str))
            
            else:
                np_valid = np.vstack((np_valid, str))

            count_time_cycles += 1
            current_str_num += 1

        return {'valid_np_train': np_train,
                'valid_np_valid': np_valid}


    @classmethod
    def out_info(self, status: bool, text: str):
        print(text)
        self.status = status





#_______________________________________________________________________________________________

    # def check_type(self, dataset : np.array) -> bool:

    #     status_log = ["Check data finished successfull", "Check data finished error"]

    #     # Удаляю строку с названиями столбцов
    #     dataset_v = np.delete(dataset, (0), axis=0)

    #     try:
    #         for st in dataset_v:
    #             for col in st:
                    
    #                 if(np.isnan(col)):
    #                     print(status_log[0])
    #                     self.status = True
    #                     return self.status
            
    #         print(status_log[0])
    #         self.status = False
    #         return self.status

    #     except:
    #         print(status_log[1])
    #         self.status = False
    #         return self.status