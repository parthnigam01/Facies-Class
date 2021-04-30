from best_model_finder import tuner
from sklearn.model_selection import train_test_split

class Stacking:
    """
                This class shall  be used to train the stacking model.
                Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

                """

    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def get_stacked(self,x,y,base_algo,algo_list):
        """
            Method Name: get_stacked
            x : Feature Columns
            y : Target Column
            base_algo : Algorithm which will be used to blend rest of algorithm
            algo_list: Base algorithms used in stacking
        """
        self.logger_object.log(self.file_object, 'Entered the get_stacked method of Stacking class')
        # spliting data into training set and holding 50 % of data
        train, val_train, test, val_test = train_test_split(x, y, test_size=0.5)

        # spliting training data into training and test data
        x_train, x_test, y_train, y_test = train_test_split(train, test, test_size=0.2)
        model_finder = tuner.Model_Finder(self.file_object, self.log_writer)  # object initialization
        # Here we are creating input data for Meta-Algo
        predict = []
        # Fitting base models with train data
        for i, algo in enumerate(algo_list):
            if algo == "rfr":
                rfr_model = model_finder.get_best_params_for_random_forest(x_train, y_train)
                #z = rfr_model.fit(x_train, y_train)
            elif algo == "lgbm":
                lgbm_model = model_finder.get_best_params_for_lgbm(x_train,y_train)
            elif algo == "svc":
                svc_model = model_finder.get_best_params_for_svc(x_train,y_train)
                #####

            filename = 'base_model_' + str(i) + '.sav'  # Saving Base models
            with open(filename, 'wb') as f:
                pickle.dump(z, f)
            # Here we are giving predictions on base models for validation data
            i = z.predict(val_train)
            predict.append(i)
            predict_val = np.column_stack(predict)  # Blending all predictions to feed meta-algo

        # Here we are creating input to check accuracy on Trained Meta-Algo
        test = []
        for i, algo in enumerate(algo_list):
            i = algo.predict(x_test)
            test.append(i)
            predict_test = np.column_stack(test)

        # Fitting meta algorithm with
        model = meta_alg.fit(predict_val, val_test)

        # Uncheck below 3 lines to apply Cross-validation
        # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        # scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        # print(score)
        print('Test Score of meta algorithm is', meta_alg.score(predict_test, y_test))
        print('Train Score of meta algorithm is', meta_alg.score(predict_val, val_test))

        # Saving Model for
        filename = 'stacking_model.sav'
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(model.accuracy())

        return model
        for algo in algo_list:
            if algo == "rfr":
                rfr_model =  model_finder.get_best_params_for_random_forest()


