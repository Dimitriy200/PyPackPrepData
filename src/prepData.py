import pandas as pd
import os
import pickle
import numpy as np

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class PrepData:


    def __init__(self):
        #11
        self.__defaultPipline = PrepData.createDefault_Pipline()
        
        # Код состояния для функций
        self.status: bool

# Вызов дефолтного Pipline - внутренняя функция
    @property
    def defaultPipline(self):
        return self.__defaultPipline
    
    
    @staticmethod
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
    

    def save_Pipeline(self, saved_pipeline: Pipeline, save_path: str) -> bool:
        status_log = ["Save pipline successfull", "Save pipline error"]

        try:
            with open(save_path, 'wb') as handle:
                save_pik_pipeline = pickle.dumps(saved_pipeline)
            
            print(status_log[0])
            self.status = True
            return self.status
        
        except:
            print(status_log[1])
            self.status = False
            return self.status


    def load_Pipeline(self, load_path: str) -> Pipeline:
        status_log = ["Load pipline successfull", "Load pipline error"]

        try:
            with open(load_path, 'rb') as handle:
                save_pik_pipeline = pickle.load(handle)

            print(status_log[0])
            self.status = True
            return save_pik_pipeline
        
        except:
            print(status_log[1])
            self.status = False
            return 0


    def fit_pipeline(self,
                     pipeline: Pipeline,
                     fit_data: np.array)-> Pipeline:
        
        status_log = ["Check data and fit pipeline successfull", "Check data and fit pipeline error"]

        while (self.check_nan_dataFrame(fit_data) == False):
            self.to_standardization_df(fit_data)
                
        self.start_fit_pipeline(pipeline, fit_data)



    def employ_Pipline(self,
                       df: pd.DataFrame,
                        inp_dir: str,       # B inpFilesList и outFilesList указывать полный путь
                        out_dir: str,
                        pipline: Pipeline = defaultPipline,) -> bool:
        
        status_log = ["Preprocess data finished successfull", "Preprocess data finished error"]

        # Получаем все документы в папке
        inpFilesList = os.listdir(inp_dir)
        outFilesList = inpFilesList
        
        # Проверяем датасет на пригодность
        if self.check_type(df):
            print("Dataset is GOOD")

        else:
            print("Dataset is BAD")
            self.status = False
            return self.status

        for fl in inpFilesList:
            print(fl)

        # На конце выходной строки дирректории должна стоять "/"
        if inp_dir[-1] != '/':
            inp_dir = f"{inp_dir}/"
        if out_dir[-1] != '/':
            out_dir = f"{out_dir}/"

        # Применение
        try:
            for file in inpFilesList:
                if file == ".gitkeep":
                    continue
                    
                print("    Processed -> ", file)

                dataFrame = pd.read_csv(inp_dir + file)
                newDataFrame = pd.DataFrame(dataFrame, columns=pipline['scaler'].get_feature_names_out(dataFrame.columns))
                
                # Сохранение
                filename, extension = os.path.splitext(file)
                newDataFrame.to_pickle(f"{out_dir}new_{filename}.pickle")
            
            print(status_log[0])
            self.status = True
            return self.status
        
        except:
            print(status_log[1])
            self.status = False
            return self.status


#           --- Service Functions --- 


    def start_fit_pipeline(self,
                            pipeline: Pipeline, 
                            fit_data: np.array) -> Pipeline:
        
        status_log = ["Fit pipline successfull", "Fit pipline error"]

        try:
            new_pipeline = pipeline.fit(fit_data)

            print(status_log[1])
            self.status = True
            return new_pipeline
        
        except:
            print(status_log[0])
            self.status = False
            return 0


    # 1 - Добавление столбца с индексами
    # 2 - Удаление 0-й строки с названиями столбцов
    # 3 - Удаление строк с пропусками
    def to_standardization_df(dataframe: np.array) -> np.array:
    
        # 1 - Добавление столбца с индексами
        str_count, col_count = dataframe.shape
        indexses = []
        indexses.append(0)

        print("str -->", str_count,
            "\ncol -->", col_count)
        
        for i in range(str_count-1):
            indexses.append(i)
        
        dataframe = np.insert(dataframe, 0, indexses, axis= 1)


        # 2 - Удаление 0-й строки с названиями столбцов
        dataframe = dataframe[1:, 0:]


        # 3 - Удаление строк с пропусками
        

        return dataframe


    def check_nan_dataFrame(dataset: np.array):

        dataset_v = np.delete(dataset, (0), axis=0)
        res: bool
        res = False

        for st in dataset_v:
            for col in st:
                # print(f"{col} --> {type(col)}")
                if(np.isnan(col)):
                    res = True
                    return res
        
        return res













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