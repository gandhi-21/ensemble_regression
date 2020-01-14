# Training the model
export TRAINING_DATA=inputs/train_encoded.csv
export TEST_DATA=inputs/test_encoded.csv
export MODEL=$1
python -m src.train