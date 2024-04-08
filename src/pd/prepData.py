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
            new_pipeline = pipeline.fit(fit_data)

            print(status_log[1])
            self.status = True
            return new_pipeline
        
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
                       df: pd.DataFrame,
                        inp_dir: str,
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

    
    def check_type(self, dataset : pd.DataFrame) -> bool:
    
        check_type1: type = type(0.0)
        check_type2: type = type(0)

        colums = dataset.columns.tolist()

        status_log = ["Check data finished successfull", "Check data finished error"]

        for col in colums:
            for var in dataset[col]:
                if check_type1 == type(var) or check_type2 == type(var):
                    continue
                
                else:
                    print(status_log[1])
                    self.status = False
                    return self.status
                
        print(status_log[0])
        self.status = True
        return self.status