# Load model 1 and model 2 and then make a prediction and average the score or something
import os
import pandas as pd
import joblib

from . import dispatcher

TEST_DATA = os.environ.get("TEST_DATA")
MODEL1 = os.environ.get("MODEL1")
MODEL2 = os.environ.get("MODEL2")

def predict():

    test_df = pd.read_csv(TEST_DATA)
    test_idx = test_df["PassengerId"].values

    test_df = test_df.drop(["PassengerId"], axis=1)

    regressor1 = joblib.load(os.path.join("models", f"{MODEL1}.pkl"))
    regressor2 = joblib.load(os.path.join("models", f"{MODEL2}.pkl"))

    # Make predictions from both and then decide on what value to take

    preds1 = regressor1.predict(test_df)
    preds2 = regressor2.predict(test_df)

    print(preds1[:5])
    print(preds2[:5])

    final_preds = (preds1 + preds2) / 2.0

    print(final_preds[:5])

    final_preds = [1 if x>=0.7 else 0 for x in final_preds]

    print(final_preds[:5])

    sub = pd.DataFrame({
        "PassengerId": test_idx,
        "Survived": final_preds
    })

    return sub


if __name__ == "__main__":
    submission = predict()
    submission.to_csv('output/submission.csv', index=False)

