import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

# Load dataset
# For GitHub/local use a relative path like "data/labeled.combined.csv"
data = pd.read_csv("data/labeled.combined.csv")

data.info()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

# (Optional, for notebooks)
# !pip install node2vec networkx

from sklearn.preprocessing import RobustScaler
exclude_columns = ['label']
excluded_data = data[exclude_columns]

scaling_data = data.drop(columns=exclude_columns)

scaler = RobustScaler()
scaled_data = scaler.fit_transform(scaling_data)

scaled_df = pd.DataFrame(scaled_data, columns=scaling_data.columns)

duplicate_columns = data.columns[data.columns.duplicated()].tolist()
if duplicate_columns:
    print(f"Duplicate column names found: {duplicate_columns}")

final_df = pd.concat([scaled_df, excluded_data.reset_index(drop=True)], axis=1)

import networkx as nx
from sklearn.neighbors import kneighbors_graph
from node2vec import Node2Vec

X_scaled = final_df.drop(columns=['label'])

# (Optional, for notebooks)
# !pip install pecanpy
# !pip show pecanpy

k = 10
print("Constructing k-NN graph...")
adj_matrix = kneighbors_graph(
    X_scaled,
    n_neighbors=k,
    mode='connectivity',
    include_self=False,
    n_jobs=-1
)
G = nx.from_scipy_sparse_array(adj_matrix)
G = nx.relabel_nodes(G, {i: str(i) for i in range(len(G.nodes))})
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

print("Generating Node2Vec embeddings...")
node2vec = Node2Vec(
    G,
    dimensions=64,
    walk_length=30,
    num_walks=5,
    p=1.0,
    q=1.0,
    workers=8,
)
model = node2vec.fit(window=5, min_count=1, batch_words=1000)
embeddings = np.array([model.wv[str(i)] for i in range(len(G.nodes))])
print("Embeddings Shape:", embeddings.shape)

np.save("node2vec_embeddings.npy", embeddings)
print("Embeddings saved to node2vec_embeddings.npy")

y = final_df['label']

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ✅ Start timer for measuring total execution time of all models
start_time = time.time()

# Define models
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
    "SVM": SVC(),
    "ADA": AdaBoostClassifier(),
    "ET": ExtraTreesClassifier(),
    "LightGBM": LGBMClassifier(),
    "DT": DecisionTreeClassifier(),
    "MLP": MLPClassifier(),
    "KNN": KNeighborsClassifier(),
    "Hist": HistGradientBoostingClassifier(),
    "RF": RandomForestClassifier(),
    "GB": GradientBoostingClassifier(),
    "LR": LogisticRegression(),
    "NB": GaussianNB(),
    "XGB": XGBClassifier(),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "SGD": SGDClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0),
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


def test_model_performance(models, X_train, X_test, y_train, y_test):
    aurocs = []
    accuracies = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
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
            y_test, predicted, average='macro'
        ) * 100
        macro_recall = recall_score(
            y_test, predicted, average='macro'
        ) * 100
        macro_f1score = f1_score(
            y_test, predicted, average='macro'
        ) * 100
        weighted_precision = precision_score(
            y_test, predicted, average='weighted'
        ) * 100
        weighted_recall = recall_score(
            y_test, predicted, average='weighted'
        ) * 100
        weighted_f1score = f1_score(
            y_test, predicted, average='weighted'
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

    # ✅ Compute mean and standard deviation of AUROC and Accuracy
    print("\n-----------------------------")
    print(f"Average AUROC: {np.mean(aurocs):.2f}% ± {np.std(aurocs):.2f}")
    print(f"Average Accuracy: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}")
    print("-----------------------------")


# Select one fold for demonstration
for train_ix, test_ix in kfold.split(embeddings, y):
    X_train, X_test = embeddings[train_ix], embeddings[test_ix]
    y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
    break  # run only one fold as an example

# Run the evaluation function
test_model_performance(models, X_train, X_test, y_train, y_test)

# ✅ End timer
end_time = time.time()
execution_time = end_time - start_time
print(f"\nTotal execution time: {execution_time:.2f} seconds")

# --- Plot ROC curves ---
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from itertools import cycle

colors = cycle([
    'crimson', 'darkorange', 'gold', 'limegreen', 'deepskyblue',
    'blueviolet', 'deeppink', 'teal', 'chocolate', 'royalblue',
    'indigo', 'forestgreen', 'salmon', 'orchid', 'darkturquoise',
    'plum', 'mediumslateblue'
])

plt.figure(figsize=(10, 8))
for name, clf in models.items():
    clf.fit(X_train, y_train)
    if hasattr(clf, "decision_function"):
        y_score = clf.decision_function(X_test)
    else:
        y_score = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    color = next(colors)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})', color=color)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Receiver Operating Characteristic (ROC) of Machine Learning Models')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import RocCurveDisplay

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_disp = RocCurveDisplay.from_estimator(rf, X_test, y_test)
plt.show()
