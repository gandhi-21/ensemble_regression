import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score
import joblib

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")

MODEL = os.environ.get("MODEL")

if __name__ == "__main__":
    
    train_df = pd.read_csv(TRAINING_DATA)

    train_x = train_df.drop(['Survived', 'PassengerId'], axis=1)
    train_y = train_df['Survived']

    # Test and Train split

    num_test = 0.3
    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=num_test, random_state=100)

    # Train the model

    regressor = dispatcher.MODELS[MODEL]
    regressor.fit(X_train, y_train)
    regressor_prediction = regressor.predict(X_test)
    regressor_score = mean_squared_error(y_test, regressor_prediction)
    print(regressor_score)
    explained_variance = explained_variance_score(y_test, regressor_prediction)
    print(explained_variance)
    
    # Save the models after training

    joblib.dump(regressor, f"models/{MODEL}.pkl")
    