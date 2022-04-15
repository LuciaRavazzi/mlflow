

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)



if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    np.random.seed(40)

    #--- Import dataset.
    csv_url = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep = ",")
    except Exception as e:
        logger.exception("File path not found.")

    #--- Split data into train and test.
    train, test = train_test_split(data)

    X_train = train.drop(["quality"], axis = 1)
    X_test = test.drop(["quality"], axis = 1)
    y_train = train[["quality"]]
    y_test = test[["quality"]]


    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 1 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha = alpha, l1_ratio = l1_ratio, random_state = 42)
        lr.fit(X_train, y_train)

        pred = lr.predict(X_test)

        rmse = mean_squared_error(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        #--- Log params.
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        #--- Log metrics.
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        #--- Log model.        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(lr, "model")           




