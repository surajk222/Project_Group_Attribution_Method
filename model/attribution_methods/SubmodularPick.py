import numpy as np

class SubmodularPick() :
    """Class for Submodular Pick
       
       It is a greedy optimization Algorithm to pick some instanes from the whole dataset to maximize
       the non redundant coverage.
    """
    
    def __init__(self, explainer, data, predict_fn, sample_size=1000, no_exps=5, num_features=16) :
        """

        Args:
            explainer (_type_): _description_
            data : An array where each datapoint is a input into predict_fn
            predict_fn : takes a tensor input and outputs prediction probabilities
            sample_size : The number of instance to be explained
            no_exps : The number of explanations returned 
            num_features : Number of features to be present in explanation
        """
        if sample_size > len(data) :
            sample_size = data
        
        self.local_explanations = []
        self.local_prediction = []
        self.prediction_score = []
        for i in range(sample_size) :
            _,local_exp,local_pred, pred_score = explainer.attribute(data.numpy()[i], predict_fn, num_features=num_features)
            self.local_explanations.append(local_exp)
            self.local_prediction.append(local_pred)
            self.prediction_score.append(pred_score)
            
        no_exps = min(int(no_exps) , len(self.local_explanations))
        
        feature_dicts = {}
        feature_count = 0
        for exp in self.local_explanations:
            labels = list(exp.keys())
            for label in labels :
                for feature,_ in exp[label] :
                    if feature not in feature_dicts.keys():
                        feature_dicts[feature] = (feature_count)
                        feature_count +=1
        d = len(feature_dicts.keys())
        W = np.zeros((len(self.local_explanations) , d))
        
        for i,exp in enumerate(self.local_explanations) :
            labels = list(exp.keys())
            for label in labels :
                for feature, value in exp[label] :
                    W[i, feature_dicts[feature]] += value
        
        imp_fn = np.sum(abs(W), axis=0)**0.5
        
        V = []
        indices = list(range(len(self.local_explanations)))
        for x in range(no_exps) :
            best = 0
            best_ind = None
            current = 0
            for i in indices:
                current = np.dot((np.sum(abs(W)[V + [i]], axis=0) > 0), imp_fn)
                if current >= best :
                    best = current
                    best_ind = i
            V.append(best_ind)
            indices.remove(i)
        
        self.sp_explanations = [self.local_explanations[i] for i in V]
        self.V = V
            