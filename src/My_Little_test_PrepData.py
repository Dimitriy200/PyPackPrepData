import PrepData as prd
import numpy as np

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



GOOD_dataFrame = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1,1,1]])

BAD_dataFrame  = np.array([[1,"", 1],
                           [1, 1, 1],
                           ["","","",]])

prepData = prd.PrepData()


simple_inputer = KNNImputer(n_neighbors = 2)
std_scaler = StandardScaler()

pipeline = Pipeline([('imputer', simple_inputer),('scaler', std_scaler)])



# 1
def test_to_standardization_df(dataFrame: np.array, 
                               prepData: prd.PrepData):
    
    prepData.to_standardization_df(dataFrame)


# 2
def test_check_nan_dataFrame(dataFrame: np.array, 
                             prepData: prd.PrepData):
    
    prepData.check_nan_dataFrame(dataFrame)


# 3
def test_save_Pipeline(saved_pipeline: Pipeline, 
                       save_path: str, 
                       prepData: prd.PrepData):
    
    prepData.save_Pipeline(saved_pipeline, save_path)


# 4
def test_load_Pipeline(load_path: str, prepData: prd.PrepData):
    prepData.load_Pipeline(load_path)


# 5
def test_fit_pipeline(pipeline: Pipeline,
                     fit_data: np.array,
                     prepData: prd.PrepData):
    
    prepData.fit_pipeline(pipeline, fit_data)


# 6
def test_employ_Pipline(dataFrame: np.array,
                        inp_dir: str,
                        out_dir: str,
                        pipline: Pipeline,
                        prepData: prd.PrepData):
    
    prepData.employ_Pipline(dataFrame, 
                            inp_dir, 
                            out_dir,
                            pipline)



test_to_standardization_df(BAD_dataFrame)

test_check_nan_dataFrame(BAD_dataFrame, prepData)

test_check_nan_dataFrame(GOOD_dataFrame, prepData)

test_save_Pipeline(pipeline, )

test_load_Pipeline()

test_fit_pipeline()

test_employ_Pipline()