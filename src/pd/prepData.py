import pandas as pd
import os

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class PrepData():


    def __init__(self):
        self.__defaultPipline = PrepData.createDefaultPipline()
        self.status: bool


    def createDefaultPipline(self):

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



    @property
    def defaultPipline(self):
        return self.__defaultPipline


    # B inpFilesList и outFilesList указывать полный путь
    def processing_data(self,
                        inpFilesList: list,
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

    

    def is_pass(dataFrame: pd.DataFrame):
        res: bool
        
        

        return res

    