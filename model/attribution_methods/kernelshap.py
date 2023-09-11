import math
import numpy as np
import torch
from numpy.random import choice, randint
from statsmodels.api import WLS
import statsmodels.api as sm


class Kernelshap():

    """
    Implementation of KernelSHAP attribution method.
    """

    def __init__(
            self,
            model,
            data,
            n = 16
        ) -> None:
        self.model = model
        self.data = data
        self.n = n
        
    # calculation of weights according to KernelSHAP weighting kernel
    def weighting_kernel(
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
            sample_percentage = 1.0

    ):
        
        data = self.data
        n = self.n

        powerset_gen = Kernelshap.powerset_generate(self)
        powerset = []
        for set in powerset_gen:
            powerset.append(set)

        # print(len(powerset))

        # sample part of subsets to reduce computational complexity
        # sampling according to weights
        if sample_percentage == 1.0:
            subset_selection = powerset
        
        else: 
            number_of_samples = round(len(powerset) * sample_percentage)
            weights = []
            for set in powerset:
                weights.append(Kernelshap.weighting_kernel(len(set)))

            # give empty and complete subset high weight
            weights[0] = 10**4
            weights[-1] = 10**4
            

            # normalize weights
            summed = sum(weights)
            weights_norm = [float(i)/summed for i in weights]
           
            powerset_range = np.arange(0, len(powerset))
            indices_selection = choice(powerset_range, number_of_samples, replace=False, p=weights_norm)
            # indices_selection = choice(powerset_range, number_of_samples)
            # print(indices_selection)

            subset_selection = []
            for selected in indices_selection:
                subset_selection.append(powerset[selected])

            # print(subset_selection)
        

        # create matrix from subset selection
        X = np.zeros((len(subset_selection), n))
        for i in range(len(subset_selection)):
            for j in range(n):
                if j in subset_selection[i]:
                    X[i][j] = 1
        X = sm.add_constant(X)

        # compute array with value function evaluations for subsets
        # y values for wls calculation
        baseline = np.zeros(n)
        y = []
        for k in range(0, len(subset_selection)):
             # tmp = Kernelshap.value_function(self, self.model, subset_selection[k], x, baseline)
             tmp = Kernelshap.value_function(self, subset_selection[k], x, 50)
             y.append(tmp)
        # print(y)
        

        # wls calculation
        # wls_model = WLS(y, X, weights=weights)
        wls_model = WLS(y, X)
        scores = wls_model.fit()
        return scores.params



    # value function with sampling for marginal features
    def value_function(
            self,
            subset,
            x, 
            number_of_samples = 10
    ):
        model = self.model
        data = self.data
        n = self.n

        indices_samples = randint(len(data), size=number_of_samples)
        expectation_value = 0

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
            output2, output_index = model.get_max_feature(tensor_values)
            # output = model(tensor_values).detach().numpy()
            # expectation_sum = np.add(expectation_sum, output)
            expectation_value += output2
        # print(expectation_sum)
        return expectation_value/number_of_samples



