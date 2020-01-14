
# Cleaning the data
# export TRAINING_DATA=inputs/train.csv
# export TEST_DATA=inputs/test.csv
# python -m src.data_cleaning

# Feature Encoding the data
# export TRAINING_DATA=inputs/train_clean.csv
# export TEST_DATA=inputs/test_clean.csv
# python -m src.feature_encoding

# Training the model
# export TRAINING_DATA=inputs/train_encoded.csv
# export TEST_DATA=inputs/test_encoded.csv
# export MODEL=$1
# python -m src.train

# Predicting using the model
# export TEST_DATA=inputs/test_encoded.csv
# export MODEL1=$1
# export MODEL2=$2
# python -m src.predict