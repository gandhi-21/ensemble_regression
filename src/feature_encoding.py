import os
import pandas as pd
from sklearn import preprocessing

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")

def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']
    df_combined = pd.concat([df_train[features], df_test[features]])

    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test

if __name__ == "__main__":

    train_df = pd.read_csv(TRAINING_DATA)
    test_df = pd.read_csv(TEST_DATA)

    df_train, df_test = encode_features(train_df, test_df)

    df_train.to_csv("inputs/train_encoded.csv", index=False)
    df_test.to_csv("inputs/test_encoded.csv", index=False)