import os
import pandas as pd
import sys
parent_directory = (os.path.abspath(os.path.join("", os.pardir)))
model_path = os.path.join(parent_directory, "submission", "models")
sys.path.append(model_path)
from ets_model import ETSModel

def predict_with_ETS():
    data_file_path = os.path.join(parent_directory, "submission", "merged_data", "allData.csv")
    df = pd.read_csv(data_file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    df.index = pd.to_datetime(df['Date'])

    #Filtering Data for training cutoff and Null values
    df = df[df.index >= '2015-01-05']
    df = df[df.index < '2023-12-01']
    #print(df['day_ahead_prices_EURO_x'])
    df = df.reset_index(drop=True)

    #Generate the ets model from the model folder in the final submissions repo
    ets = ETSModel("ets_model", "ETS_modell")
    y_train = df['day_ahead_prices_EURO_x']
    if y_train.isnull().any():
            print("Warning: Missing values will be passed in y_train. The missing values are: ")
            missing_values_in_column = df[df['day_ahead_prices_EURO_x'].isnull()]
            print(missing_values_in_column)
    X_train = df["Date"]
    ets.train(X_train, y_train)

    #Create a prediction
    X = pd.DataFrame({'Date': pd.date_range('2023-12-01 00:00:00', periods=100, freq='h')})
   # print(X)
    results = ets.predict(X)
    #print(results)
    sys.stdout.write(str(results))

def main():
    predict_with_ETS()

if __name__ == "__main__":
    main()