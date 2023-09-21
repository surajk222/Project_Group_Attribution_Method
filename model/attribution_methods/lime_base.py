import numpy as np
import scipy as sp
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state

class LimeBase :
    """Class for learning a locally linear sparse model from perturbed data
    """
    def __init__(self,
                 kernel_fn,
                 random_state = None) :
        """Init function

        Args:
            kernel_fn : function that transforms a numpy array into an array of proximity values
            random_state : Generate randome numbers. Defaults to None.
        """
        
        self.kernel_fn = kernel_fn
        self.random_state = check_random_state(random_state)
    

    
    def feature_selection(self, X_vector, Y_vector, weights, num_features, method):
        """Selects features for the model. See instance explanation  to understand the parameters
        """
        if method == 'none':
            return np.array(range(X_vector.shape[1]))
        elif method == 'lasso_path':
            weighted_X_vector = ((X_vector - np.average(X_vector, axis=0, weights=weights)) * np.sqrt(weights[:,np.newaxis]))
            weighted_Y_vector = ((Y_vector - np.average(Y_vector, weights=weights)) * np.sqrt(weights))
            nonzero_features = np.empty(weighted_X_vector.shape[1])
            alphas,_,coefs = lars_path(weighted_X_vector, weighted_Y_vector, method='lasso', verbose=False)
            
            for i in range(len(coefs.T) - 1, 0, -1) :
                nonzero_features = coefs.T[i].nonzero()[0]
                if len(nonzero_features) <= num_features :
                    break
            return nonzero_features
    
    def instance_explanation(self, sample_data, sample_labels, distances, label, num_features, feature_selection='lasso_path', model=None) :
        """Takes the perturbed data, distances to the sample datapoints and explains the single instance

        Args:
            sample_data : Perturbed data
            sample_labels : corresponding perturbed labels
            distances : distances of the sample data from the original datapoint
            label : label for which we want an explanation
            num_features : number of features we want in an explanation
            feature_selection : How to select features to explain the instance the locally.
            model : The model chosen for explanation.
        """
        
        weights = self.kernel_fn(distances)
        used_features = self.feature_selection(sample_data, sample_labels[:,label], weights, num_features, feature_selection)
        
        if model is None :
            model = Ridge(alpha=1.0, fit_intercept=True , random_state=self.random_state)
        
        model.fit(sample_data[:,used_features], sample_labels[:,label], sample_weight=weights)
        prediction_score = model.score(sample_data[:,used_features], sample_labels[:,label], sample_weight=weights)
        local_prediction = model.predict(sample_data[0,used_features].reshape(1,-1))
        
        #print("Intercept" , model.intercept_)
        print("Local prediction" , local_prediction)
        print("Sample columns ", sample_labels[0,label])
        
        return model.intercept_,sorted(zip(used_features, model.coef_), key = lambda x : np.abs(x[1])), prediction_score, local_prediction
         
            
            
        