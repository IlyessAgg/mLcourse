# Notes

> Notes of different teachings.

## Feature Processing

* **Numerical Features **

  Scaling and Rank for numeric features — Huge impact on non-tree based models.

  Most used preprocessing techniques:

  * MinMaxScaler - to [0,1]
  * StandardScaler - to mean=0 & std=1
  * Rank - sets spaces between sorted values to be equal
  * np.log(1+x) and np.sqrt(1+x)

  Feature generation usually depends on `prior knowledge` & `data analysis`.

* **Categorical Features**

  Most used preprocessing techniques:

  * LabelEncoder()
  * OrdinalEncoder()
  * One_Hot_Encoding()

  Less used but very interesting is **Mean Encoding** — `probability of your target variable, conditional on each value of the feature.`

  Helps tremendously with tree based models.

## Ensembling

**Ensemble Modelling** — `combining many different machine learning models in order to get a more powerful prediction.`

There are various ensemble methods :

* **Averaging** : Multiple techniques exist. We can simply average the predictions of multiple models (**Blending**), or give more importance to some models over others (**Weighted Averaging**) or pick a model based on a condition on the data (**Conditional Averaging**).

* **Bagging** : Averaging slightly different versions of the same model (_example : RandomForest_). Helps avoid **overfitting**.

  *Paramaters that control bagging* : *Seed, Row/Column sampling or bootstraping, Shuffling, Number of bags etc...*

* **Boosting** : a form of weighted averaging of models where each model is built sequentially via taking into account the past model performance.

* **Stacking** : making predictions of a number of models in a hold-out set and then using a different (**Meta**) model to train on these predictions.