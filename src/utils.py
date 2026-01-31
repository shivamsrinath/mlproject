import os 
import sys
from src.exception import CustomException
import dill

import numpy as np 
import pandas as pd 
from sklearn.model_selection import GridSearchCV
from  sklearn.metrics import r2_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
 
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():

            
            if model_name == "CatBoosting Regressor":
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
                report[model_name] = r2_score(y_test, y_test_pred)
                models[model_name] = model
                continue

            
            gs = GridSearchCV(
                model,
                param_grid=param[model_name],
                cv=3,
                scoring="r2",
                error_score="raise"
            )

            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_test_pred = best_model.predict(X_test)
            report[model_name] = r2_score(y_test, y_test_pred)

            
            models[model_name] = best_model

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)


    
