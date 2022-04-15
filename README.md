
# mlflow

According to https://mlflow.org/docs/0.5.1/tutorial.html, mlflow is useful for different purposes:

- **MLFlow Tracking: Log kpi's values, parameters and artifacts (model)**.
      The idea is to build a machine learning pipeline as always. In additition, mlflow.start_run() is
      executed to monitor and store if some hyperparameters should be stored. Especially, this purpose is accomplished
      with mlflow.log_param(), mlflow.log_metric() methods e sklearn models with mlflow.sklearn.log_model(). 
      In the same directory of the script, with the command *python script.py* the folder *mlruns* is created and contains all the stored information with the kpi's and the model (pickle format). 
      Furthermore, in order to visualize the performance of the model, it's possible to execute
      _mlflow ui_ in the same folder of the script and here http://localhost:5000 the results are shown. Click the date and all the information
      about the model will be described.

- **MLFlow Projects: Package the code**.
      Here the idea is to pack a project in order to be used in cloud or by other people. 
      Inside the folder in which there is the code, add a *MLProject* file defining the conda env and the entry point.
      Remember that the *name* parameter is the folder in which the code is inside. Moreover, the *conda.yaml* file collects
      the information regarding the dependencies of the package.  
      In order to run this project, exit from the folder of the script and run: *mlflow run Notebook -P alpha=0.42*. After running this command, MLflow will run your training code in a new Conda environment with the dependencies specified in conda.yaml.   
      In this way, people don't need to touch or modify the code in order to make it run because the package containes
      all dependencies and all the necessary code to be executed. A folder *mlruns* is created to store the model and all the other information. 

- **MLFlow Models and Server: Managing and deploying models**. 
      After the execution of the previous step, a model is ready to be deployed. Inside the *mlruns* folder of the final model, 
      the file *MLmodel* is a metadata file that tells MLflow how to load the model, while *model.pkl* is a serialized version of the model that you trained.
      You can deploy that model through a local rest API or in production.
      For the first task, run mlflow models serve -m ./mlruns/0/7c1a0d5c42844dcdb8f5191146925174/artifacts/model -p 1234
      and, in order to make prediction, curl -X POST -H "Content-Type:application/json; format=pandas-split" --data "{\"columns\":[\"alcohol\", \"chlorides\", \"citric acid\", \"density\", \"fixed acidity\", \"free sulfur dioxide\", \"pH\", \"residual sugar\", \"sulphates\", \"total sulfur dioxide\", \"volatile acidity\"],\"data\":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}" http://127.0.0.1:1234/invocations.
      On the other hand, in order to put the model in production, a docker-image should be created.


 
