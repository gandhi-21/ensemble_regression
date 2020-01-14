Practice code for ensemble methods

Make a new conda environment using the following command:
conda env create --file requirements.txt

Data source:
https://www.kaggle.com/c/titanic/data

Process
1. The data is cleaned using data_cleaning.py
2. The data is feature encoded using feature_encoder.py
3. The models are trained using predict.py
4. The dispatcher handles what model to be used
5. The predictions on the test set are made using predict.py

All the models are stored in models/

Instructions to run the code and make a submission 

1. Using the requirements.txt create a new conda/pip environment
2. Activate the environment
3. Follow the instructions below to run everything using run.sh script

Instructions to run run.sh
Go to terminal and type the following commands
Make sure that the environmnet you just created is activated

data_cleaning
1. run clean_data.sh

feature_encoding
1. run feature_encode.sh

training
1. run train.sh
Models to be trained can be found in dispatcher.py

predicting
1.  run predict.sh "model1" "model2"
For this example the models used are randomForest and extraTrees
Terminal command to use:
sh predict.sh extraTrees randomForest
Models to be used for prediction can be found in dispatcher.py
