import numpy as np
import pandas as pd
# Importing dataset and preprocessing routines
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# Base classifier models:
from sklearn.linear_model import Perceptron
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble.voting_classifier import VotingClassifier


from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from des_new import DESNEW
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC

rng = np.random.RandomState(42)

acute = "cardiovascular.csv"
acu= pd.read_csv(acute, header=None, delimiter=";")
X = acu.loc[:,0:9].values
y = acu[10].values

#scaler_x = StandardScaler().fit(X)
#X = scaler_x.transform(X)
#lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
#model = SelectFromModel(lsvc, prefit=True)
#X = model.transform(X)


scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
X = SelectKBest(chi2, k=5).fit_transform(X, y)
results = []
res = []
mean_acc = []
mean_gmean = []
mean_prec = []
mean_recall = []
mean_f1 = []

classifiers = [
    KNeighborsClassifier(3),
    KNeighborsClassifier(5),
    SVC(kernel="linear", C=0.025),
    SVC(kernel="linear", C=1),
    SVC(gamma=2, C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    #GaussianNB(),
    #QuadraticDiscriminantAnalysis()
    ]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rng)

cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
for train_index, dsel_index in cv.split(X_train, y_train):
    #print("Train Index: ", train_index, "\n")
    #print("Val Index: ", dsel_index)

    X_train, X_dsel, y_train, y_dsel = X[train_index], X[dsel_index], y[train_index], y[dsel_index]
    desnew = DESNEW(classifiers, X_train, y_train, X_dsel, y_dsel, mode=3, neighbourhood=13)
    result = desnew.return_metrics()
    #res = desnew.fit_predict_return_simple(X_test, y_test)
    #print(res)
    if result == 0:
        pass
    else:
        results.append(result)
print("Stratified K-Fold: ", len(results))
for i in range(0, len(results)):
    mean_acc.append(results[i]['acc'])
    mean_f1.append(results[i]['f1'])
    mean_gmean.append(results[i]['gmean'])
    mean_prec.append(results[i]['prec'])
    mean_recall.append(results[i]['recall'])
#for acc, f1, gmean, prec, recall in results:
#accs, f1s, gmeans, precs, recs = zip(*results)
std_acc = np.std(mean_acc)
mean_acc = np.mean(mean_acc)
std_f1 = np.std(mean_f1)
mean_f1 = np.mean(mean_f1)
std_gmean = np.std(mean_gmean)
mean_gmean = np.mean(mean_gmean)
std_prec = np.std(mean_prec)
mean_prec = np.mean(mean_prec)
std_recall = np.std(mean_recall)
mean_recall = np.mean(mean_recall)

print("accuracy: %0.2f (+/- %0.2f)" % (mean_acc, std_acc))
print("gmean: %0.2f (+/- %0.2f)" % ( mean_gmean, std_gmean))
print("precision: %0.2f (+/- %0.2f)" % (mean_prec, std_prec))
print("recall: %0.2f (+/- %0.2f)" % (mean_recall, std_recall))
print("f1: %0.2f (+/- %0.2f)" % (mean_f1, std_f1))
