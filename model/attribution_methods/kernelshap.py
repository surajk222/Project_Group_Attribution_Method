import math
import numpy as np
import torch
from numpy.random import choice, randint
from statsmodels.api import WLS, OLS
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

        powerset_gen = self.powerset_generate()
        powerset = []
        for set in powerset_gen:
            powerset.append(set)
        self.powerset = powerset

        weights = []
        for set in powerset:
            weights.append(self.weighting_kernel(len(set)))
        # give high weight to empty and full subset to ensure selection
        weights[0] = 10**4
        weights[-1] = 10**4
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
            subset_percentage = 0.005,
            number_of_samples = 200,
            **kwargs
    ):
        
        data = self.data
        n = self.n

        # sample part of subsets to reduce computational complexity
        # sampling according to weights
        if subset_percentage == 1.0:
            subset_selection = self.powerset
        
        else: 
            number_of_samples = round(len(self.powerset) * subset_percentage)

            powerset_range = np.arange(0, len(self.powerset))
            indices_selection = choice(powerset_range, number_of_samples, replace=False, p=self.weights_norm)
            # indices_selection = choice(powerset_range, number_of_samples)
            # print(indices_selection)

            subset_selection = []
            for selected in indices_selection:
                subset_selection.append(self.powerset[selected])

            # print(subset_selection)

        # create matrix from subset selection
        U = np.zeros((len(subset_selection), n))
        for i in range(len(subset_selection)):
            for j in range(n+1):
                if j in subset_selection[i]:
                    U[i][j-1] = 1
        U = sm.add_constant(U)

        # compute array with value function evaluations for subsets
        # y values for wls calculation
        v = []
        for k in range(0, len(subset_selection)):
            exp = self.value_function(subset_selection[k], x, number_of_samples)
            # exp = self.value_function_zero(subset_selection[k], x)
            if target_label_index != None:
                value = exp[target_label_index]
            else: 
                value = max(exp)
                target_label_index = np.argmax(exp)
            v.append(value)
        # print(y)

        # print(self.powerset[0])
        # print(self.value_function(self.powerset[0], x, number_of_samples=number_of_samples))
        # print(self.powerset[-1])
        # print(self.value_function(self.powerset[-1], x, number_of_samples=number_of_samples))

        # wls calculation
        # wls_model = WLS(v, U, weights=weights)
        wls_model = WLS(v, U)
        scores = wls_model.fit()
        # print(scores.summary())
        # print(torch.from_numpy(scores.params[1:]))
        return torch.from_numpy(scores.params), target_label_index




    # value function with sampling for marginal features
    def value_function(
            self,
            subset,
            x, 
            target_label_index = None,
            number_of_samples = 50
    ):
        model = self.model
        data = self.data
        n = self.n

        indices_samples = randint(len(data), size=number_of_samples)
        expectation_values = np.zeros(7)
        # expectation_value = 0

        for index in indices_samples:
            instance_sample, instance_y = data[index]
            # print(instance_sample)
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
        exp = expectation_values/number_of_samples
        if subset == []:
            print("[]: ")
            print(exp)
        elif subset == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
            print("full: ")
            print(exp)
        return exp
    

    def value_function_zero(
            self,
            subset,
            x
    ):
        model = self.model
        n = self.n

        input_values = np.array([])
                
        for i in range(n):
            if (i+1) in subset:
                input_values = np.append(input_values, x[i])
            else:
                input_values = np.append(input_values, 0)
        # print(input_values)
        tensor_values = torch.from_numpy(input_values).float()
        model.eval()
        exp = model.predict(tensor_values)

        return exp
