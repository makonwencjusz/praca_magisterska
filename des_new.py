from deslib.des import DESKNN

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np

class DESNEW:
    def __init__(self, pool_classifiers, X_train, y_train, X_test, y_test, mode=0, neighbourhood=13):
        self.mode = mode
        self.neighbourhood = neighbourhood
        self.pool_classifiers = pool_classifiers
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_dsel = []
        self.y_dsel = []
        self.desknn = DESKNN(pool_classifiers) # dodaj parametry desknn na sztywno albo przez konstrutkor


    def return_metrics(self):
        
        if self.mode == 0:
            self._fit_base_classifiers(self.X_train, self.y_train)
            metrics = self.fit_predict_return_simple(self.X_test, self.y_test)

        elif self.mode == 1:
            X_train1, y_train1 = SMOTE(random_state=42).fit_resample(self.X_train, self.y_train)
            self._fit_base_classifiers(X_train1, y_train1)
            metrics = self.fit_predict_return_simple(self.X_test, self.y_test)
            
        elif self.mode == 2:
            self._fit_base_classifiers(self.X_train, self.y_train)
            self._test_sample(self.X_test, self.y_test)
            metrics = self.fit_predict_return_score(self.X_dsel, self.y_dsel)
        else:
            X_train1, y_train1 = SMOTE(random_state=42).fit_resample(self.X_train, self.y_train)
            self._fit_base_classifiers(X_train1, y_train1)
            self._test_sample(self.X_test, self.y_test)
            metrics = self.fit_predict_return_score(self.X_dsel, self.y_dsel)
        return metrics
        

    def fit_predict_return_simple(self, X_test, y_test):
        metrics_names = ['acc', 'f1', 'gmean', 'prec', 'recall']
        self.desknn.fit(X_test, y_test)
        y_pred = self.desknn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        gmean = geometric_mean_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        metrics_results = dict(zip(metrics_names, [acc, f1, gmean, prec, recall]))
        return metrics_results

    def fit_predict_return_score(self, X_dsel, y_dsel):
        predictions = []
        metrics = []
        metrics_names = ['acc', 'f1', 'gmean', 'prec', 'recall']
        
        for X_d, y_d in zip(X_dsel, y_dsel):
            self.desknn.fit(X_d, y_d)
            y_pred = self.desknn.predict(X_d)
            predictions.append(y_pred)
            acc = accuracy_score(y_d, y_pred)
            f1 = f1_score(y_d, y_pred)
            gmean = geometric_mean_score(y_d, y_pred)
            prec = precision_score(y_d, y_pred)
            recall = recall_score(y_d, y_pred)
            metrics_results = dict(zip(metrics_names, [acc, f1, gmean, prec, recall]))
            metrics.append(metrics_results)
        
        #for y_pred in predictions:
            #acc = accuracy_score(y_dsel, y_pred)
            #f1 = f1_score(y_dsel, y_pred)
            #gmean = geometric_mean_score(y_test, y_pred)
            #prec = precision_score(y_test, y_pred)
            #recall = recall_score(y_test, y_pred)
            
            #metrics_results = dict(zip(metrics_names, [acc, f1, gmean, prec, recall]))
            #metrics.append(metrics_results)
            
        gmeans = []

        for metrics_results in metrics:
            gmeans.append(metrics_results['gmean'])

        if not gmeans:
            metrics = 0
            return metrics
        else:
            idx = np.argmax(gmeans)
            return metrics[idx]
    
    
    def _test_sample(self, X_test, y_test):
        
        # sampling
        # binary classification
        unique_vals, counts = np.unique(y_test, return_counts=True)
        minority_class = unique_vals[np.argmin(counts)]
        majority_class = unique_vals[np.argmax(counts)]
        minority_class_indices = np.where(y_test == minority_class)
        minority_samples = X_test[minority_class_indices] #instancje zbioru testowego o klasie mniejszosciowej
        
        neigh = NearestNeighbors(n_neighbors=self.neighbourhood, metric="euclidean")
        neigh.fit(X_test) #n najblizszych sasiadow na calym zbiorze testowym
        
        #for sample in minority_samples:
        distances, indices = neigh.kneighbors(minority_samples)
            
        for i in range(0, len(indices)):
                # min distance for ratio
                # zbior do SMOTE
            y_i = indices[i]
            uni_val, cnts = np.unique(y_test[y_i], return_counts=True)
            if cnts[0]<13:
                if cnts[minority_class]>2 and cnts[minority_class] < cnts[majority_class]:
                    for j in range(1,len(y_i)):
                
                        if y_test[j] == minority_class:
                            distance = distances[i][j]
                            dist_func = lambda x: 1.5*x / (x + 1)
                            ratio = dist_func(distance)
                            if ratio<0.75:
                                sm = SMOTE(sampling_strategy=0.75, random_state=42,k_neighbors=2)
                            elif ratio>1:
                                sm = SMOTE(sampling_strategy=1, random_state=42, k_neighbors=2)
                            else:
                                sm = SMOTE(sampling_strategy=ratio, random_state=42, k_neighbors=2)

                            X_resampled, y_resampled = sm.fit_resample(X_test[y_i], y_test[y_i])

                            self.X_dsel.append(X_resampled)
                            self.y_dsel.append(y_resampled)
                            break
                else:
                    self.X_dsel.append(X_test[y_i])
                    self.y_dsel.append(y_test[y_i])
                

                
        
    def estimate_distance(self):
        #check only samples from minor class / classes
        #for given k neighbours (k>10) compute the distance (euclidean, Mahalonobis) from nearest same class sample
        #sample the region using SMOTE, computing density of sampling by homographic function f(x)=x/x+1 according to distance from given sample
        #if no other minor class instance inside the region -> break, cause of noise possibility
        #in other cases -> compute precision, recall, g-mean and f1 score
        pass
    
    def _fit_base_classifiers(self, X_train, y_train):
        # sampling
        for clf in self.pool_classifiers:
            clf.fit(X_train, y_train)
            