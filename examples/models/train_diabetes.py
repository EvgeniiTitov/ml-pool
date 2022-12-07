import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    # load data
    df = pd.read_csv("pima-indians-diabetes.data.csv", header=None)

    # split data into X and y
    X = df.iloc[:, 0:8]
    Y = df.iloc[:, 8]

    # split data into train and test sets
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=7
    )

    # fit model no training data
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # make predictions for test data
    print("Test features:")
    print(type(X_test))

    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    model.save_model("diabetes_xgb.json")
