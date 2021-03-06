from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import sys
sys.path.append("../")

class LinearSupportVectorClassifier():
    def __init__(self,training_data_x,training_data_y):
        self.training_data_x = training_data_x
        self.training_data_y = training_data_y

    def initialize_classifier(self,tol=0.0001,C=1.0):
        dual_val = True
        if len(self.training_data_x) > len(self.training_data_x[0]):  # if number of samples > number of features
            dual_val = False
        self.classifier = LinearSVC(dual=dual_val,tol=tol,C=C)

    def train(self):
        self.classifier.fit(self.training_data_x,self.training_data_y)

    def predict(self,sample_x):
        return self.classifier.predict(sample_x)

    def get_f1_measure(self,sample_x,sample_y):
        predictions = self.predict(sample_x)
        score = f1_score(
            y_true=sample_y,
            y_pred=predictions,
            average="micro"
        )
        return score

    def find_best_params(self,n_jobs=1,param_grid=[],cv=5):
        if not param_grid:
            param_grid = self.get_param_grid_default()


        gs = GridSearchCV(
            estimator=LinearSVC(),
            scoring='accuracy',
            param_grid=param_grid,
            n_jobs=n_jobs,
            cv=cv,
        )

        gs.fit(self.training_data_x,self.training_data_y)

        best_params = gs.best_params_
        best_score = gs.best_score_
        results = gs.cv_results_
        return best_params,best_score,results


    def get_param_grid_default(self):
        dual_val = True
        if len(self.training_data_x) > len(self.training_data_x[0]): #if number of samples > number of features
            dual_val = False

        tol_vals = []
        current_tol = 0.00000001
        while(current_tol<0.0001):
            tol_vals.append(current_tol)
            current_tol *= 10

        c_vals = []
        current_c_val = 0.001
        while(current_c_val<=100):
            c_vals.append(current_c_val)
            current_c_val *= 10

        param_grid = {"dual": [dual_val],
                      "tol" : tol_vals,
                      "C" : c_vals
                      }
        return param_grid