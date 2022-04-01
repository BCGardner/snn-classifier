# Datasets

| Name      | Description                         | Link |
| --------- | ----------------------------------- | ---- |
| iris      | Scikit-learn demo dataset           | N/A  |
| wisconsin | UCI dataset (1996)                  | https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data |
| wisc      | UCI dataset as used in Bohte (1992) | https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data |
| wisc_nmf  | NMF decomposition on wisconsin      | N/A  |

Unless otherwise stated, the datasets were benchmarked against an scikit-learn MLP, setup as follows:
- activation='logistic'
- hidden_layer_sizes=(100,)
- solver='SGD'
- batch_size=150
- max_iter=4000
- random_state=1

## iris

- Contains 150 samples, 4 features per sample.
- MLP gives a training accuracy of 98 %, after 2398 iterations.
- Accuracy worse after normalising data.

## wisconsin

- Contains 569 samples, 30 features per sample.
- MLP gives a training accuracy of 97.1 % after normalisation, after 419 iterations.
- Normalising this dataset gives better accuracy.
- Using NMF decomposition (5 components / reduced features) gives a training accuracy of 92.4 % after 744 iterations.
- NMF decomposition and then normalisation gives an accuracy of 94.0 % after 1125 iterations.

## wisc

- Contains 683 samples, 9 features per sample.
- 16 samples were removed due to missing values (original size was 699 samples).
- Likely the exact same dataset used by Bohte et al. 2002 (although Bohte included samples containing missing values).
- MLP gives a training accuracy of 97.1 % after 419 iterations.
- Normalisation makes no difference to final training accuracy.

## wisc_nmf

- Contains 683 samples, 5 features per sample.
- This dataset is derived from wisc_original: decomposed using sklearn.decomposition.NMF.
- The NMF model was initialised with a random state of 0 and five components (features).
- MLP gives a training accuracy of 96.9 % after normalisation, after 539 iterations.
- Normalising this dataset gives better accuracy.
- Using an MLP containing 30 hidden neurons gives an accuracy of 96.8 % after normalisation, after 601 iterations.
