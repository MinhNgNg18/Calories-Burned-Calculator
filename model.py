import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from xgboost import XGBRegressor
import pickle


def remove_outliers(df, col):
    Q3 = np.quantile(df[col], 0.75)
    Q1 = np.quantile(df[col], 0.25)
    IQR = Q3-Q1
    upper_bound = Q3+1.5*IQR
    lower_bound = Q1-1.5*IQR
    return df[(df[col] <= upper_bound) & (df[col] >= lower_bound)]


def preprocess():
    df = pd.read_csv('gym_members_exercise_tracking.csv')
    int_cols = [f for f in df.columns if df[f].dtype != 'O']
    for col in int_cols:
        df = remove_outliers(df, col)

    cols_to_encode = ['Gender', 'Workout_Type']
    one_hot = OneHotEncoder(drop='first', sparse_output=False)
    one_hot_encoded = one_hot.fit_transform(df[cols_to_encode])
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot.get_feature_names_out(cols_to_encode),
                                      index=df.index)
    df = df.drop(cols_to_encode, axis=1)
    df = pd.concat([df, one_hot_encoded_df], axis=1)
    return df, one_hot


def build_model():
    df, one_hot = preprocess()
    X = df.drop('Calories_Burned', axis=1)
    y = df['Calories_Burned']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    xgb = XGBRegressor()
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    grid_search = GridSearchCV(estimator=xgb,
                               param_grid=param_grid,
                               scoring=scorer,
                               cv=5,
                               verbose=1,
                               n_jobs=-1)

    grid_search.fit(X_scaled, y)
    model = grid_search.best_estimator_
    return model, scaler, one_hot


def main():
    model, scaler, one_hot = build_model()
    data = {'model': model, 'scaler': scaler, 'one_hot': one_hot}
    with open('saved_steps.pkl', 'wb') as file:
        pickle.dump(data, file)


if __name__ == '__main__':
    main()


