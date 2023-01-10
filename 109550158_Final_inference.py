import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
from sklearn.metrics import r2_score, roc_auc_score
from feature_engine.encoding import WoEEncoder
from sklearn.linear_model import LogisticRegression, HuberRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import NearMiss, ClusterCentroids
from imblearn.over_sampling import SMOTE
import joblib

t_list = ['product_code', 'loading', 'attribute_0', 'measurement_0', 'measurement_1', 'measurement_2', 'measurement_3', 'measurement_4', 'measurement_5', 'measurement_6', 'measurement_7', 'measurement_8',
          'measurement_9', 'measurement_10', 'measurement_11', 'measurement_12', 'measurement_13', 'measurement_14', 'measurement_15', 'measurement_16', 'measurement_17', 'failure']

if __name__ == "__main__":
    # load test data
    test = pd.read_csv("data/tabular-playground-series-aug-2022/test.csv")
    test['attribute_0'] = test['attribute_0'].replace(['material_7'], 1.0)
    test['attribute_0'] = test['attribute_0'].replace(['material_5'], 0.0)

    # KNN imputer
    all_columns = list(test.columns)
    test = test.drop(list(set(all_columns) - set(t_list)), axis=1)
    for code in test.product_code.unique():
        knnimputer1 = KNNImputer(n_neighbors=100)
        test.loc[test.product_code == code, t_list[1:-1]] = knnimputer1.fit_transform(
            test.loc[test.product_code == code, t_list[1:-1]])
        condition = test["product_code"] == code
        for c_n in t_list:
            if c_n != 'product_code' and c_n != 'failure' and c_n != 'attribute_0':
                test.where(condition)[c_n] = (test.where(condition)[
                    c_n] - test.where(condition)[c_n].mean()) / test.where(condition)[c_n].std()
    
    with open('my_data/my_columns.npy', 'rb') as f:
        my_columns = np.load(f, allow_pickle=True)
    my_columns = my_columns[my_columns != 'failure']
    folds_dict = {f'Fold 1': [['C', 'D', 'E'], ['A', 'B']],
                  'Fold 2': [['B', 'D', 'E'], ['A', 'C']],
                  'Fold 3': [['B', 'C', 'E'], ['A', 'D']],
                  'Fold 4': [['B', 'C', 'D'], ['A', 'E']],
                  'Fold 5': [['A', 'D', 'E'], ['B', 'C']],
                  'Fold 6': [['A', 'C', 'E'], ['B', 'D']],
                  'Fold 7': [['A', 'C', 'D'], ['B', 'E']],
                  'Fold 8': [['A', 'B', 'E'], ['C', 'D']],
                  'Fold 9': [['A', 'B', 'D'], ['C', 'E']],
                  'Fold 10': [['A', 'B', 'C'], ['D', 'E']]}

    test_predictions = np.zeros((test.shape[0], 1))
    model_idx = 0
    # test
    for fold in folds_dict.keys():
        model_lr = joblib.load('models/LR_model'+str(model_idx))
        model_idx += 1
        test_pred = model_lr.predict_proba(test[my_columns].values)[
            :, 1].reshape(-1, 1)
        test_predictions += test_pred / 10

    submission = pd.read_csv(
        'data/tabular-playground-series-aug-2022/sample_submission.csv')
    submission['failure'] = test_predictions
    submission['failure'] = submission['failure'].rank(pct=True).values
    submission.to_csv('109550158.csv', index=False)
