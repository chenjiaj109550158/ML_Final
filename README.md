# ML_Final

## 1. Dependencies

Python 3.8 is required.

Clone the repository first
```
git clone https://github.com/chenjiaj109550158/ML_Final.git
```
and then install the dependencies.
```
pip install -r requirements_examples.txt
```

## 2. Training section
Run training section
```
python 109550158_Final_train.py
```
This will produce my_columns.npy in my_data and LR_models in models.
my_columns.npy is the attributes chosen as features.
LR_models are the train models.

There are some hyperparameters that can be adjusted to get better(or worse) performance.

n_neighbors:
n_neighbors represents the number of the neighbors in KNN imputer.
```python
    for code in train.product_code.unique():
        knnimputer1 = KNNImputer(n_neighbors=100)
        train.loc[train.product_code == code, t_list[1:-1]] = knnimputer1.fit_transform(
            train.loc[train.product_code == code, t_list[1:-1]])
        condition = train["product_code"] == code
        for c_n in t_list:
            if c_n != 'product_code' and c_n != 'failure' and c_n != 'attribute_0':
                train.where(condition)[c_n] = (train.where(condition)[
                    c_n] - train.where(condition)[c_n].mean()) / train.where(condition)[c_n].std()
```

my_columns_n and 'attribute_0'
my_columns_n represents the number of the attributes chosen as features.
It's optional to choose 'attribute_0' as a feature.
```python
    my_columns_n = 2
    my_columns = np.array(cor.index)[0:1+my_columns_n]
    my_columns = np.append(my_columns, 'attribute_0')
```

random_state:
The random_state of SMOTE
```python
sm = SMOTE(random_state=42, n_jobs=-1)
```

max_iter and C:
Hyperparameter in LogisticRegression
```python
        model_lr = LogisticRegression(
            max_iter=700, C=0.1, dual=False, penalty="l2", solver='newton-cg')
```

## 3. Evaluation section
Run Evaluation section
```
python 109550158_Final_inference.py
```




