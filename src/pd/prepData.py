import pandas as pd
import os
import pickle

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class PrepData():


    def __init__(self):
        self.__defaultPipline = PrepData.createDefaultPipline()
        
        # Код состояния для функций
        self.status: bool


    def createDefault_Pipline(self):
        status_log = ["Create pipline successfull", "Create pipline error"]

        try:
            simple_inputer = KNNImputer(n_neighbors = 2)
            std_scaler = StandardScaler()
            pipe_num = Pipeline([('imputer', simple_inputer), ('scaler', std_scaler)])

            print(status_log[0])
            return pipe_num
        
        except:
            print(status_log[1])
            return 0
    

    def save_Pipeline(self, saved_pipeline: Pipeline, save_path: str):
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


    def fit_pipeline(self, pipeline: Pipeline, fit_data: pd.DataFrame) -> Pipeline:
        status_log = ["Fit pipline successfull", "Fit pipline error"]

        try:
            pipeline.fit(X = fit_data)

            print(status_log[1])
            self.status = True
            return pipeline
        except:
            print(status_log[0])
            self.status = False
            return 0


    # Вызов дефолтного Pipline - внутренняя функция
    @property
    def defaultPipline(self):
        return self.__defaultPipline


    # B inpFilesList и outFilesList указывать полный путь
    def employ_Pipline(self,
                        inpFilesList: list, # путь к папке ()
                        outFilesList: list,
                        pipline: Pipeline = defaultPipline):
        
        status_log = ["Preprocess data finished successfull", "Preprocess data finished error"]

        try:
            for url in range(len(inpFilesList)):
                dataFrame = pd.read_csv(url)
                newDataFrame = pipline.fit_transform(dataFrame)
                newDataFrame = pd.DataFrame(newDataFrame, columns=pipline['scaler'].get_feature_names_out(dataFrame.columns))
                newDataFrame.to_pickle(outFilesList[url])
            
            print(status_log[0])
            self.status = True
            return self.status
        
        except:
            print(status_log[1])
            self.status = False
            return self.status

    

    def is_null(dataFrame: pd.DataFrame):
        res: bool
        
        

        return res