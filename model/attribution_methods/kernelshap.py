import math
import numpy as np
import torch
from numpy.random import choice, randint
from statsmodels.api import WLS, OLS
import statsmodels.api as sm
from model.util.baseline_generator import generate_uniform_baseline
from model.training import train_autobaseline


class Kernelshap():

    """
    Implementation of KernelSHAP attribution method.
    """

    def __init__(
            self,
            model,
            data,
            n = 16,
        ):

        """
        Args:
            model: a model to evaluate the 
            data: training data for evaluations in value function
            n: number of features

        Returns:
            
        """

        self.model = model
        self.data = data
        self.n = n

        # create powerset of features
        powerset_gen = self.powerset_generate()
        powerset = []
        for set in powerset_gen:
            powerset.append(set)
        self.powerset = powerset[1:-1]

        weights = []
        for set in self.powerset:
            weights.append(self.weighting_kernel(len(set)))
        self.weights = weights
        # normalize weights
        summed = sum(self.weights)
        weights_norm = [float(i)/summed for i in self.weights]
        self.weights_norm = weights_norm
        

    # calculation of weights according to KernelSHAP weighting kernel
    def weighting_kernel(
            self,
            subset_size,
            n: int = 16,
    ) -> float:
        return (n - 1) / math.comb(n, subset_size) * subset_size * (n - subset_size)
   

    # generates powerset of given n elements
    def powerset_generate(
                self
        ):
        n = self.n
        powerset_size = 2**n

        featureset = []
        for k in range(1, n+1):
            featureset.append(k)

        for i in range(0, powerset_size):
            powerset = []
            for j in range(0, n):
                if ((i & (1 << j)) > 0):
                    powerset.append(featureset[j])
            yield powerset



    # actual attribution score computation
    def attribute(
            self, 
            x,
            target_label_index = None,
            subset_percentage = 0.01,
            number_of_samples = 50,
            log_odds: bool = True,
            set_seed: bool = False,
            attribution_baseline = np.zeros(16),
            **kwargs
    ):
        
        if target_label_index == None:
            target_label_index = self.model.predict(x).argmax().item()

        data = self.data
        n = self.n

        # sample part of subsets to reduce computational complexity
        # sampling according to weights
        if subset_percentage == 1.0:
            subset_selection = self.powerset
        
        else: 
            number_of_subsets = round(len(self.powerset) * subset_percentage)

            powerset_range = np.arange(0, len(self.powerset))
            if set_seed:
                np.random.seed(43)
        
            # select random samples 
            indices_selection = np.random.choice(powerset_range, number_of_subsets, replace=False, p=self.weights_norm)
            # indices_selection = choice(powerset_range, number_of_subsets, replace=False)
            # if 0 not in indices_selection:
            #     indices_selection = np.append(indices_selection, 0)
            # if (len(self.powerset)-1) not in indices_selection:
            #     indices_selection = np.append(indices_selection, (len(self.powerset)-1))
            # indices_selection = choice(powerset_range, number_of_samples)
            # print(indices_selection)
            

            subset_selection = []
            weights_subsets = []
            for selected in indices_selection:
                subset_selection.append(self.powerset[selected])
                weights_subsets.append(self.weights_norm[selected])
            if [] not in subset_selection:
                subset_selection.append([])
            if [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] not in subset_selection:
                subset_selection.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

            # print(subset_selection)

        # create matrix from subset selection
        U = np.zeros((len(subset_selection), n))
        for i in range(len(subset_selection)):
            for j in range(n+1):
                if j in subset_selection[i]:
                    U[i][j-1] = 1
        U = sm.add_constant(U)

        # create weights for wls
        wls_weights = np.ones(len(subset_selection))
        for i in range(len(subset_selection)):
            subset = subset_selection[i]
            if subset == []:
                wls_weights[i] = 10**5
            elif subset == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
                wls_weights[i] = 10**5

        wls_weights_norm = [float(i)/sum(wls_weights) for i in wls_weights]

        if set_seed:
            np.random.seed(43)

        indices_samples = np.random.randint(len(data), size=number_of_samples)

        # compute array with value function evaluations for subsets
        # y values for wls calculation
        v = []
        testing = 0
        for k in range(0, len(subset_selection)):
            testing += 1
            exp = self.value_function(subset_selection[k], x, indices_samples=indices_samples, number_of_samples=number_of_samples)
            # exp = self.value_function_baseline(subset_selection[k], x, baseline=attribution_baseline)
            value = exp[target_label_index]
            # if subset_selection[k] == []:
                # print(value)
            v.append(value)


        # wls calculation
        wls_model = WLS(v, U, weights=wls_weights_norm)
        scores = wls_model.fit()

        if log_odds:
            return torch.from_numpy(scores.params[1:]), target_label_index
        else:
            return torch.from_numpy(scores.params), target_label_index


    # value function with sampling for marginal features
    def value_function(
            self,
            subset,
            x, 
            indices_samples,
            target_label_index = None,
            number_of_samples = 50
    ):
        model = self.model
        data = self.data
        n = self.n

        # np.random.seed(43)
        # indices_samples = np.random.randint(len(data), size=number_of_samples)
        # print(indices_samples)
        expectation_values = np.zeros(7)
        # expectation_value = 0

        for index in indices_samples:
            instance_sample, instance_y = data[index]
            input_values = np.array([])
            for i in range(n):
                if (i+1) in subset:
                    input_values = np.append(input_values, x[i])
                else:
                    input_values = np.append(input_values, instance_sample[i])
            # print(input_values)
            tensor_values = torch.from_numpy(input_values).float()
            model.eval()
            # output, output_index = model.get_max_value(tensor_values)
            output2 = model.predict(tensor_values)
            # output = model(tensor_values).detach().numpy()
            expectation_values = np.add(expectation_values, output2)
            # expectation_value += output
        # print(expectation_values)
        exp = expectation_values/len(indices_samples)
        # if subset == []:
        #     print("[]: ")
        #     print(exp)
        # elif subset == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
        #     print("full: ")
        #     print(exp)
        return exp
    
    # value function with given baseline
    # access to this function has to be changed manually in kernelshap class
    def value_function_baseline(
            self,
            subset,
            x,
            baseline
    ):
        model = self.model
        n = self.n

        input_values = np.array([])
                
        for i in range(n):
            if (i+1) in subset:
                input_values = np.append(input_values, x[i])
            else:
                input_values = np.append(input_values, baseline[i])
        # print(input_values)
        tensor_values = torch.from_numpy(input_values).float()
        model.eval()
        exp = model.predict(tensor_values)

        return exp
    
    
    def get_mean_prediction(
            self
    ):
        expectation = np.zeros(7)
        for i in range(len(self.data)):
            x, y = self.data[i]
            res = self.model.predict(x)
            expectation = np.add(expectation, res)
        mean_prediction = expectation / len(self.data)
        # print(mean_prediction)
        return mean_prediction
    


