import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

# Start global timer
start_time = time.time()

# Load dataset (use relative path for GitHub/local)
data = pd.read_csv("data/labeled.combined.csv")

data.info()
data['label'].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

num_columns = data.columns


def summary(dataframe):
    print(f"data shape: {dataframe.shape}")

    summ = pd.DataFrame(
        columns=[
            "dtype",
            "missing",
            "missing[%]",
            "unique",
            "min",
            "max",
            "median",
            "std",
            "outliers",
            "lower_bound",
            "upper_bound",
        ]
    )

    for col in num_columns:
        summ.loc[col, "dtype"] = dataframe[col].dtype
        summ.loc[col, "missing"] = dataframe[col].isnull().sum()
        summ.loc[col, "missing[%]"] = (
            dataframe[col].isnull().sum() / len(dataframe) * 100
        )
        summ.loc[col, "unique"] = dataframe[col].nunique()
        summ.loc[col, "min"] = dataframe[col].min()
        summ.loc[col, "max"] = dataframe[col].max()
        summ.loc[col, "median"] = dataframe[col].median()
        summ.loc[col, "std"] = dataframe[col].std()

        q1 = dataframe[col].quantile(0.25)
        q3 = dataframe[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = dataframe[
            (dataframe[col] < lower_bound) | (dataframe[col] > upper_bound)
        ][col]
        summ.loc[col, "outliers"] = outliers.count()
        summ.loc[col, "lower_bound"] = lower_bound
        summ.loc[col, "upper_bound"] = upper_bound

    return summ


summary(data)
data.head()

target = data["label"].value_counts()
target.plot.pie()

from sklearn.preprocessing import RobustScaler

exclude_columns = ["label"]
excluded_data = data[exclude_columns]

scaling_data = data.drop(columns=exclude_columns)

scaler = RobustScaler()
scaled_data = scaler.fit_transform(scaling_data)

scaled_df = pd.DataFrame(scaled_data, columns=scaling_data.columns)

duplicate_columns = data.columns[data.columns.duplicated()].tolist()
if duplicate_columns:
    print(f"Duplicate column names found: {duplicate_columns}")

final_df = pd.concat(
    [scaled_df, excluded_data.reset_index(drop=True)],
    axis=1,
)

final_df.head()
summary(final_df)

X = final_df.drop("label", axis=1)
y = final_df["label"]

from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_ix, test_ix in kfold.split(X, y):
    X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
    y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
    break  # use only the first fold

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

y_train.value_counts()

# (Optional for notebooks)
# !pip install ctgan
from ctgan import CTGAN

cat_feature = ["label"]
train = pd.concat([X_train, y_train], axis=1)
train["label"].value_counts()

ctgan = CTGAN(verbose=True)
# ctgan.fit(train, cat_feature, epochs=200)

# Use a relative path instead of /content for GitHub/local usage
loaded = ctgan.load("models/js_train.pkl")

samples = loaded.sample(16800)
samples = samples[samples["label"] == 0]
ctgan_result_df = pd.concat([train, samples])
ctgan_result_df.head()

ctgan_result_df["label"].value_counts()

X_train_gan = ctgan_result_df.drop(["label"], axis=1)
y_train_gan = ctgan_result_df["label"]

# (Optional for notebooks)
# !pip install catboost
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)

models = {
    "SVM": SVC(C=500, kernel="poly", degree=5, gamma="auto"),
    "ADA": AdaBoostClassifier(n_estimators=500, learning_rate=0.05),
    "ET": ExtraTreesClassifier(
        n_estimators=500,
        max_depth=5,
        random_state=42,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=500,
        learning_rate=0.1,
        num_leaves=5,
        max_depth=5,
    ),
    "DT": DecisionTreeClassifier(
        max_depth=2,
        min_samples_split=100,
        random_state=42,
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(1000,),
        activation="relu",
        solver="adam",
        max_iter=200,
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=50,
        algorithm="brute",
        p=1,
    ),
    "Hist": HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=3,
        learning_rate=0.01,
        min_samples_leaf=100,
    ),
    "RF": RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        random_state=42,
    ),
    "GB": GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
    ),
    "LR": LogisticRegression(C=1000, max_iter=500),
    "NB": GaussianNB(),
    "XGB": XGBClassifier(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.01,
    ),
    "LDA": LinearDiscriminantAnalysis(
        solver="lsqr",
        shrinkage="auto",
    ),
    "QDA": QuadraticDiscriminantAnalysis(reg_param=1.0),
    "SGD": SGDClassifier(
        max_iter=500,
        loss="hinge",
        penalty="l2",
    ),
    "CatBoost": CatBoostClassifier(
        iterations=100,
        depth=3,
        learning_rate=0.01,
        l2_leaf_reg=10,
        verbose=0,
    ),
}

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    matthews_corrcoef,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
)
from imblearn.metrics import specificity_score, geometric_mean_score


def test_model_performance(models, X_train_gan, X_test, y_train_gan, y_test):
    # Lists to store AUROC and Accuracy for computing mean and std
    aurocs = []
    accuracies = []

    for model_name, model in models.items():
        model.fit(X_train_gan, y_train_gan)
        predicted = model.predict(X_test)
        roc_auc = roc_auc_score(y_test, predicted) * 100
        accuracy = accuracy_score(y_test, predicted) * 100
        aurocs.append(roc_auc)
        accuracies.append(accuracy)

        sensivity = recall_score(y_test, predicted) * 100
        precision = precision_score(y_test, predicted) * 100
        f1score = f1_score(y_test, predicted) * 100
        specificity = specificity_score(y_test, predicted) * 100
        mcc = matthews_corrcoef(y_test, predicted) * 100
        kappa = cohen_kappa_score(y_test, predicted) * 100
        gmeans = geometric_mean_score(y_test, predicted) * 100
        macro_precision = precision_score(
            y_test, predicted, average="macro"
        ) * 100
        macro_recall = recall_score(
            y_test, predicted, average="macro"
        ) * 100
        macro_f1score = f1_score(
            y_test, predicted, average="macro"
        ) * 100
        weighted_precision = precision_score(
            y_test, predicted, average="weighted"
        ) * 100
        weighted_recall = recall_score(
            y_test, predicted, average="weighted"
        ) * 100
        weighted_f1score = f1_score(
            y_test, predicted, average="weighted"
        ) * 100

        print(
            f'{model_name}: AUROC - {roc_auc:.2f}%, Accuracy - {accuracy:.2f}%, '
            f'Recall - {sensivity:.2f}%, Precision - {precision:.2f}%, '
            f'F1_Score - {f1score:.2f}%, Specificity - {specificity:.2f}%, '
            f'MCC - {mcc:.2f}%, Kappa - {kappa:.2f}%, '
            f'Geometric Mean - {gmeans:.2f}%, Macro Precision - {macro_precision:.2f}%, '
            f'Macro Recall - {macro_recall:.2f}%, Macro F1 Score - {macro_f1score:.2f}%, '
            f'Weighted Precision - {weighted_precision:.2f}%, '
            f'Weighted Recall - {weighted_recall:.2f}%, '
            f'Weighted F1 Score - {weighted_f1score:.2f}%'
        )

    # Print mean and standard deviation of AUROC and Accuracy
    print("\n-----------------------------")
    print(f"Average AUROC: {np.mean(aurocs):.2f}% ± {np.std(aurocs):.2f}")
    print(f"Average Accuracy: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}")
    print("-----------------------------")


test_model_performance(models, X_train_gan, X_test, y_train_gan, y_test)

# End global timer
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from itertools import cycle

colors = cycle(
    [
        "crimson",
        "darkorange",
        "gold",
        "limegreen",
        "deepskyblue",
        "blueviolet",
        "deeppink",
        "teal",
        "chocolate",
        "royalblue",
        "indigo",
        "forestgreen",
        "salmon",
        "orchid",
        "darkturquoise",
        "plum",
        "mediumslateblue",
    ]
)

plt.figure(figsize=(10, 8))

for name, clf in models.items():
    clf.fit(X_train_gan, y_train_gan)
    if hasattr(clf, "decision_function"):
        y_score = clf.decision_function(X_test)
    else:
        y_score = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    color = next(colors)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})", color=color)

plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Receiver Operating Characteristic (ROC) of Machine Learning Models")
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import RocCurveDisplay

lgbm = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.1,
    num_leaves=5,
    max_depth=5,
)
lgbm.fit(X_train_gan, y_train_gan)
lgbm_disp = RocCurveDisplay.from_estimator(lgbm, X_test, y_test)
plt.show()
