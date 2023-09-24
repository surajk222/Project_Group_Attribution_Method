import numpy as np
import torch
import scipy as sp
import sklearn
import warnings
from scipy import sparse
import collections
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
import copy as cp
from functools import partial
from model.attribution_methods import lime_base


class Lime() :
    """Implementation of LIME Explainer from Marco, Sameer, Carlos
    """
    def __init__(self,
                training_data,
                mode="classification",
                feature_names = None,
                categorical_features = None,
                categorical_names = None,
                kernel_width = None,
                kernel = None,
                class_names = None,
                random_state = None,
                sample_around_instance = False) :
        """Args :

        Args:
            training_data : numpy 2d array
            mode : classification or regression
            feature_names : list of names corresponding to the names of the colmuns in the training data
            categorical_features : categorical columns indices
            categorical_names : categorical names[x][y] represents the name of the yth value of column x
            kernel_width : exponential kernel
            kernel : similarity kernel that takes euclidean distances and kernel width as input and output weights in (0,1)
            class_names : list of class names, ordered according to whatever the classifier is using. If not present will be 0, 1, ...
            random_state : Randomstate will be used to generate random numbers
            sample_around_instance : if True, will sampe continuous features in perturbed samples from a normal centered at the instance being explained.
                Otherwise the normal is centered on the mean of the feature data.
        """
        self.random_state = check_random_state(random_state)
        self.mode = mode
        self.categorical_names = categorical_names
        self.sample_around_instance = sample_around_instance
        
        if categorical_features is None :
            categorical_features = []
        if feature_names is None :
            feature_names = [str(i) for i in range(training_data.shape[1])]
        
        self.categorical_features = list(categorical_features)
        self.feature_names = list(feature_names)
        
        if kernel_width is None :
            kernel_width = np.sqrt(training_data.shape[1]) * 0.75
        kernel_width = float(kernel_width)
        
        if kernel is None :
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d**2) / kernel_width ** 2))
        
        kernel_fn = partial(kernel, kernel_width=kernel_width)
        
        self.base = lime_base.LimeBase(kernel_fn, random_state = self.random_state)
        self.class_names = class_names
        
        self.feature_values = {}
        self.feature_fre = {}
        
        for x in range(training_data.shape[1]):
            col = training_data[:,x]
            
            count_features = collections.Counter(col)
            values, freq = map(list, zip(*(sorted(count_features.items()))))
            
            self.feature_values[x] = values
            self.feature_fre[x] = (np.array(freq)/float(np.sum(freq)))
            
            
        
    def attribute(self, instance, predict_fn, labels=(1,), num_features = 10, num_samples = 5000, model=None, sampling_method = 'gaussian') :
        """Generate explanations for a prediction

        Args:
            instance : single instance to explained (row data)
            predict : takes a numpy array as input and gives the prediction probabilities as output
            labels : iterable to be explained
            num_features : Number of features to be inculded in explanation.
            num_samples : size of the synthetic data to learn the linear model
            model : model to be used in the explanation
            sampling_method : Method to perturbed data
        """
        interpreted_data, orginial_data = self.perturbation(instance, num_samples, sampling_method)
        

        distances = pairwise_distances(interpreted_data, interpreted_data[0].reshape(1, -1)).ravel()
        
        n = predict_fn(torch.from_numpy((orginial_data)).float()).numpy()
        
        if self.mode == 'classification' :
            if len(n.shape) == 1 :
                raise NotImplementedError("LIME does not support classifier models without probability scores")
            elif len(n.shape) == 2 :
                if self.class_names is None :
                    for i in range(n[0].shape[0]) :
                        self.class_names.append(str(i))
                else:
                    self.class_names = list(self.class_names)
                if not np.allclose(n.sum(axis=1), 1.0) :
                    raise ValueError("Prediction probabilites doesn't sum up to 1")
            else:
                raise ValueError("The model outputs some unknow dimensions")
        
        feature_names = cp.deepcopy(self.feature_names)
        if feature_names is None :
            feature_names = [str(i) for i in range(instance.shape[1])]
        
        for i in self.categorical_features :
            name = int(instance[i])
            if i in self.categorical_names :
                name = self.categorical_names[i][name]
            feature_names[i] = feature_names[i] + name
        
        labels = np.argsort(n[0])[-1:]
        intercept = {}
        local_exp = {}
        score = {}
        local_pred = {}
        for label in labels :
            (intercept[label], local_exp[label], score[label], local_pred[label]) = self.base.instance_explanation(interpreted_data,
                                                                                                                   n,
                                                                                                                   distances,
                                                                                                                   label,num_features,model=model,feature_selection='lasso_path')
        
        return intercept, local_exp, score, local_pred
    
    def perturbation(self,
                     data,
                     num_samples,
                     sampling_method):
        """Neighbourdhood data point around the prediction

        Args:
            data : 1d numpy array, corresponding to a row
            num_samples : number of samples to learn the sparse linear model
            sampling_method : gaussian
            
        Returns :
        A tuple(interpreted data, original data)
        interpredted data : data with interpretable representations encoded with 0 or 1
        original data : same as the original data
        """
        
        n_cols = data.shape[0]
        interpredted_data = np.zeros((num_samples,n_cols))
        
        
        if sampling_method == 'gaussian' :
            interpredted_data = self.random_state.normal(0,1, num_samples * n_cols).reshape(num_samples,n_cols)
            interpredted_data = np.array(interpredted_data)
        else :
            warnings.warn('''Invalid input for sampling method ''')
            interpredted_data = self.random_state.normal(0,1, num_samples * n_cols).reshape(num_samples,n_cols)
            interpredted_data = np.array(interpredted_data)
        
        
        interpredted_data[0] = data.copy()
        original_data = interpredted_data.copy()
        
        for col in range(16) :
            values = self.feature_values[col]
            freqs = self.feature_fre[col]
            
            original_col = self.random_state.choice(values , size=num_samples, replace=True, p=freqs)
            bin_col = (original_col == data[col]).astype(int)
            bin_col[0] = 1
            original_col[0] = interpredted_data[0,col]
            interpredted_data[:,col] = bin_col
            original_data[:,col] = original_col
        
        original_data[0] = data
        return interpredted_data, original_data
    
        