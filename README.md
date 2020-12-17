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

In this project we are going to look at ways we can make the insurance claims process more efficient. Efficiencies in insurance claims severity analysis can help provide suitable insurance packages to customers and provide targeted assistance to better serve them. Finally, we would like to build a model that can predict severity of claims so as to improve the claims service to ensure a worry-free customer experience.

Using a dataset from Kaggle; provided by AllState, a US-based insurance company; the training dataset consists of 130 attributes (features) and the loss value for each observation. The dataset contains 188,318 observations where each row represents an insurance claim. This means each claim is a process that requires 130 different information. So, the main questions are:

1) Do we require all these attributes/information?
2) Can we eliminate any of these attributes to be more efficient?
  If yes, then:
  - 2.1) which continuous variable are least important and can be dropped?
  - 2.2) which categorical values is least important and can be dropped?
3) Which attributes are most important for AllState?
4) Finally, can we create an algorithm to predict claims severity?

## Methods Used <a name="method"></a>
- Exploratory data analysis to understand the Allstate insurance claim dataset
- Feature selection and elimination using Correlation, Constant Variance and Chi-Square statistical tests
- Use PCA and Feature Importances to find the most important features
- Understanding ensemble Machine Learning algorithms 
- Hyper-parameter tuning using Scikit-Learn functions
- Model selection using RMSE as the model evaluation metric

## File Descriptions <a name="files"></a>
1. `Insurance_severity_claims.ipynb` : Notebook containing the whole project combined including EDA & Machine learning model
2. `API` : folder containing 3 files : 
                   1. `claimsPrediction_model_API.py` : flask API code for deployment       
                   2. `columns_to_drop.csv`  : csv files containing features to be dropped and is used in the API file
                   3. `tunedmodel_rf` : pickle file containing the RandomForest model used for prediction
3. `dataset.zip` :zipped folder containing the train and test datasets
4. `requirements.txt` : text file containing the required  libraries & packages to execute the code


## Results<a name="results"></a>
- I was able to drop the number of features from 130 to 39 and I trained a ML algorithm which work quite was and was able to make prediction
- The prediction using the test dataset was submitted on Kaggle and a score of 3011.62 was achieved which can be improved.

More information about the project and the main findings of the code can be found at the post available [here](https://fbhugaloo.medium.com/predicting-claims-severity-a-machine-learning-approach-e6744760d04c)

## Installation <a name="installation"></a>
- To clone the repository use: git clone https://github.com/fardil-b/Insurance-Claims-Severity-Prediction.git

- The code should run with no issues using Python versions 3.0 and above. The additional libraries required to execute the code can be installed using `pip` with `pip install -r requirements.txt`


## Licensing, Authors, Acknowledgements<a name="licensing"></a>
Must give credit to AllState for the data.  You can find the Licensing for the data and other descriptive information at the Kaggle link available [here](https://www.kaggle.com/c/allstate-claims-severity/data). Otherwise, feel free to use the code here as you would like! 
