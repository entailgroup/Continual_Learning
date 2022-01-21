import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def read_EPCOR(data_root, force_reload=False):
    try:
        with open(data_root+"loaded_dataset.pk", 'rb') as f:
            df = pickle.load(f)
    except:
        force_reload = True

    if force_reload:
        df = pd.read_csv(data_root+"cons_data.csv")
        df["YEAR"] = df.apply(lambda x: x["DATE"]//10000, axis=1)
        df["MONTH"] = df.apply(lambda x: (x["DATE"]//100)%100, axis=1)
        df["DAY"] = df.apply(lambda x: x["DATE"]%100, axis=1)
        
        df.drop("DATE", axis=1)
        df = df[["SITE_ID", "RATE_CLASS", "YEAR", "MONTH", "DAY", "HOUR_ENDING", "IS_DAYLIGHT_SAVING", "CONSUMPTION_KWH"]]
        df = df.sort_values(["SITE_ID", "YEAR", "MONTH", "DAY", "HOUR_ENDING"], ascending=True)

        with open(data_root+"loaded_dataset.pk", 'wb') as f:
            pickle.dump(df, f)

    return df

def preprocess_EPCOR(df, train_prop, lookback):
    train_size = int(np.ceil(df.shape[0] * train_prop))

    # TODO: try (0.1, 1) scale
    sc = MinMaxScaler()
    features = ["MONTH", "DAY", "HOUR_ENDING"]
    df[features] = sc.fit_transform(df[features])
    df["IS_DAYLIGHT_SAVING"] = df["IS_DAYLIGHT_SAVING"].astype(int)
    df["YEAR"] = df["YEAR"] - 2018
    one_hot = pd.get_dummies(df["RATE_CLASS"])
    df = df.drop("RATE_CLASS", axis=1)
    df = df.join(one_hot)

    data = df.to_numpy()

    inputs = []
    labels = []

    for i in range(lookback, len(data)):
        if len(np.unique(data[i-lookback:i, 0])) == 1:
            
            inputs.append(data[i-lookback:i,1:])
            labels.append(data[i,-1])

    inputs = np.array(inputs)
    labels = np.array(labels).reshape(-1,1)

    X_train = inputs[:train_size]
    y_train = labels[:train_size]

    X_test = inputs[train_size:]
    y_test = labels[train_size:]

    return (X_train, y_train), (X_test, y_test)