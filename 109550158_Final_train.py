import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

t_list = ['product_code', 'loading', 'attribute_0', 'measurement_0', 'measurement_1', 'measurement_2', 'measurement_3', 'measurement_4', 'measurement_5', 'measurement_6', 'measurement_7', 'measurement_8',
          'measurement_9', 'measurement_10', 'measurement_11', 'measurement_12', 'measurement_13', 'measurement_14', 'measurement_15', 'measurement_16', 'measurement_17', 'failure']

if __name__ == "__main__":
    # load train data
    train = pd.read_csv("data/tabular-playground-series-aug-2022/train.csv")
    train['attribute_0'] = train['attribute_0'].replace(['material_7'], 1.0)
    train['attribute_0'] = train['attribute_0'].replace(['material_5'], 0.0)

    # KNN imputer
    all_columns = list(train.columns)
    train = train.drop(list(set(all_columns) - set(t_list)), axis=1)
    for code in train.product_code.unique():
        knnimputer1 = KNNImputer(n_neighbors=100)
        train.loc[train.product_code == code, t_list[1:-1]] = knnimputer1.fit_transform(
            train.loc[train.product_code == code, t_list[1:-1]])
        condition = train["product_code"] == code
        for c_n in t_list:
            if c_n != 'product_code' and c_n != 'failure' and c_n != 'attribute_0':
                train.where(condition)[c_n] = (train.where(condition)[
                    c_n] - train.where(condition)[c_n].mean()) / train.where(condition)[c_n].std()

    # calculate attributes correlation
    my_columns = np.array(t_list)
    my_columns = my_columns[my_columns != 'product_code']
    cor = np.absolute(train.drop(['product_code'], axis=1).corr())[
        'failure'].sort_values(ascending=False)
    my_columns_n = 2
    my_columns = np.array(cor.index)[0:1+my_columns_n]
    my_columns = np.append(my_columns, 'attribute_0')
    with open('my_data/my_columns.npy', 'wb') as f:
        np.save(f, my_columns)

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

    oof_predictions = np.zeros((train.shape[0], 1))
    oof_targets = np.zeros((train.shape[0]))

    auc_score = []
    targets = []
    model_idx = 0

    # training
    sm = SMOTE(random_state=42, n_jobs=-1)
    for fold in folds_dict.keys():
        x_train, y_train = train[train['product_code'].isin(
            folds_dict[fold][0])][my_columns].values, train[train['product_code'].isin(folds_dict[fold][0])]['failure'].values
        x_valid, y_valid = train[train['product_code'].isin(
            folds_dict[fold][1])][my_columns].values, train[train['product_code'].isin(folds_dict[fold][1])]['failure'].values
        trn_index = train[train['product_code'].isin(
            folds_dict[fold][0])][my_columns].index
        vld_index = train[train['product_code'].isin(
            folds_dict[fold][1])][my_columns].index

        x_train, y_train = sm.fit_resample(x_train, y_train)
        model_lr = LogisticRegression(
            max_iter=700, C=0.1, dual=False, penalty="l2", solver='newton-cg')
        model_lr.fit(x_train, y_train)
        joblib.dump(model_lr, 'models/LR_model'+str(model_idx))
        model_idx += 1

        predictions = model_lr.predict_proba(x_valid)[:, 1].reshape(-1, 1)
        oof_predictions[vld_index] = predictions
        score = roc_auc_score(y_valid, predictions)
        auc_score.append(score)
        oof_targets[vld_index] = y_valid

        print(f'AUC score {fold} : {score}')
        print()

    print()
    print(f'Method1 CV AUC score: {np.mean(np.array(auc_score))}')
    print('Method2 CV AUC score: ', roc_auc_score(oof_targets, oof_predictions))
