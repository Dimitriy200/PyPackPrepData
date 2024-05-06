import pandas as pd
import os
import pickle
import numpy as np

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
                        inp_dir: str,       # B inpFilesList и outFilesList указывать полный путь
                        out_dir: str,
                        pipline: Pipeline = defaultPipline) -> bool:
        
        status_log  =           ["Preprocess data finished successfull",        "Preprocess data finished error"]
        get_doc_log =           ["Getting a list of documents...",              "Documents have been received"]

        # На конце выходной строки дирректории должна стоять "/"
        if inp_dir[-1] != '/':
            inp_dir = f"{inp_dir}/"
        if out_dir[-1] != '/':
            out_dir = f"{out_dir}/"

        # Получаем все документы в папке
        print(get_doc_log[0])
        inpFilesList = os.listdir(inp_dir)
        outFilesList = inpFilesList
        
        # Вывод списка файлов в диррекории
        for fl in inpFilesList:
            print("   .../" + fl)
            
        print(get_doc_log[1])

        # Применение
        for file in inpFilesList:
            if file == ".gitkeep":
                continue
            
            print("  Processed --> ", file)

            # Загружаем i-тый фаил
            dataFrame = genfromtxt(inp_dir + file, delimiter = ',')
            
            # Проверяем датасет на пригодность (отсутствие пропусков)
            for line in dataFrame:
                if (self.is_nan_dataFrame_Line(line) == False):
                    continue
                else:
                    print("Dataset is BAD Starting standartization dataframe...")
                    self.to_standardization_df(dataFrame)
                    break
            
            print("Dataset is GOOD  Starting employ pipeline...")
            
            # newDataFrame = pd.DataFrame(dataFrame, columns = pipline['scaler'].get_feature_names_out(dataFrame.columns))
            newDataFrame = pipline.fit_transform(dataFrame)

            # Сохранение
            fileName = out_dir + "New_" + file + ".csv"
            np.savetxt(fileName, newDataFrame, delimiter = ",")
        
        self.out_info(True, status_log[0])
        return True


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
                print(f"res_dataFrame = {res_dataFrame}")
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
                self.out_info(self.status, status_log[0] + f" result is {self.status}")
                return self.status
            
        self.out_info(self.status, status_log[0] + f" result is {self.status}")
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