# Insurance-Claims-Severity-Prediction
How severe is an insurance claim?


### Table of Contents

1.  [Project Motivation](#motivation)
2.  [Methods Used](#method)
3. [File Descriptions](#files)
4. [Results](#results)
2. [Installation](#installation)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Motivation<a name="motivation"></a>
Allstate, an US-based insurance company is developing automated methods to predict claims severity in order to provide better claims service for Allstate's customers. The goal is to build a model that can help Allstate to predict the severity of the claims accurately and  finally, create an API and deploy the model to generate the prediction at runtime. 

## Methods Used <a name="method"></a>
- Exploratory data analysis to understand the Allstate insurance claim dataset
- Feature selection and elimination using Correlation, Constant Variance and Chi-Square statistical tests
- Understanding ensemble Machine Learning algorithms 
- Hyper-parameter tuning using Scikit-Learn functions
- Model selection using RMSE as the model evaluation metric
- Model deployment creating FlaskAPI

## File Descriptions <a name="files"></a>
1. `Insurance_severity_claims.ipynb` : Notebook containing the whole project combined including EDA & Machine learning model
2. `API` : folder containing 3 files : 
                   1. `claimsPrediction_model_API.py` : flask API code for deployment       
                   2. `columns_to_drop.csv`  : csv files containing features to be dropped and is used in the API file
                   3. `tunedmodel_rf` : pickle file containing the RandomForest model used for prediction
3. `dataset.zip` :zipped folder containing the train and test datasets
4. `requirements.txt` : text file containing the required  libraries & packages to execute the code


## Results<a name="results"></a>
- The prediction using the test dataset was submitted on Kaggle and a score of 3011.62 was achieved which can be improved.
- We were also successfully able to deploy the model and get prediction. But need to improve the model for greater accuracy.
![image](https://user-images.githubusercontent.com/61830624/102217579-1242e300-3edd-11eb-8e8d-3aa3aba87bb8.png)

More information about the project and the main findings of the code can be found at the post available [here](https://fbhugaloo.medium.com/predicting-claims-severity-a-machine-learning-approach-e6744760d04c)

## Installation <a name="installation"></a>
- To clone the repository use: git clone https://github.com/fardil-b/Insurance-Claims-Severity-Prediction.git

- The code should run with no issues using Python versions 3.0 and above. The additional libraries required to execute the code can be installed using `pip` with `pip install -r requirements.txt`


## Licensing, Authors, Acknowledgements<a name="licensing"></a>
Must give credit to AllState for the data.  You can find the Licensing for the data and other descriptive information at the Kaggle link available [here](https://www.kaggle.com/c/allstate-claims-severity/data). Otherwise, feel free to use the code here as you would like! 
