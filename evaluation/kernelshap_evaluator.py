import numpy as np
from model.attribution_methods import kernelshap
from numpy.random import randint
from evaluation.utils.lineplot import visualize_scores
import torch

class KernelshapEvaluator():
    """
    KernelSHAP Evaluator
    """

    def __init__(self, model, data, explainer:kernelshap.Kernelshap):
        self.model = model
        self.data = data
        self.explainer = explainer


    def sample_data(
            self,
            testsize
    ): 
        np.random.seed(25)
        test_indices = np.random.randint(len(self.data), size=testsize)
        return test_indices


    def evaluate_sample_size(
            self,
            low,
            high,
            step_size = 10
    ):
        test_indices = self.sample_data(1)
        print(test_indices)
        count = int(np.ceil((high-low)/step_size))
        results = np.zeros((count, 17))
        for i in test_indices:
            x, y = self.data[i]
            target_label_index = self.model.predict(x).argmax().item()
            index = 0
            for j in range(low, high, step_size):
                x, y = self.data[i]
                scores = self.explainer.attribute(x, target_label_index=target_label_index, number_of_samples=j, subset_percentage=0.01, set_seed=True, log_odds=False)
                # print(scores)
                # sample_array = np.append(sample_array, scores[0])
                for k in range(17):
                    results[index][k] = scores[0][k]
                index += 1
        change = self.compute_change_between_levels(results)
        visualize_scores(results, change, 'Veränderung der Attribution Scores \n bei variierender \n Anzahl der Instanzen', 'sample_size', 'Anzahl der Instanzen', low, high, step_size)
        # print(np.var(results))
        self.compute_change_between_levels(results)


    def evaluate_subset_percentage(
            self,
            low,
            high,
            step_size = 0.001
    ): 
        test_indices = self.sample_data(1)
        print(test_indices)
        count = int(np.ceil((high-low)/step_size))
        # print(count)
        results = np.zeros((count, 17))
        for i in test_indices:
            # subset_array = np.array([])
            x, y = self.data[i]
            target_label_index = self.model.predict(x).argmax().item()
            index = 0
            for j in np.arange(low, high, step_size):
                scores = self.explainer.attribute(x, target_label_index=target_label_index, subset_percentage=j, number_of_samples=50, set_seed=True, log_odds=False)
                # subset_array = np.append(subset_array, scores[0])
                for k in range(17):
                    results[index][k] = scores[0][k]
                index += 1
                # print(results)
        change = self.compute_change_between_levels(results)
        visualize_scores(results, change, 'Veränderung der Attribution Scores \n bei variierendem \n Anteil der betrachteten Teilmengen', 'subset_percentage', 'Anteil der Teilmengen', low, high, step_size)


    def evaluate_both(
            self
    ):
        subset_percentage = [0.005, 0.01, 0.015, 0.02, 0.025]
        sample_size = [100, 200, 300, 400, 500]
        test_indices = self.sample_data(1)
        results = np.zeros((5, 17))
        for i in test_indices:
            print(i)
            count = 0
            for j in np.arange(5):
                x, y = self.data[i]
                scores = self.explainer.attribute(x, subset_percentage=subset_percentage[j], number_of_samples=sample_size[j], set_seed=True, log_odds=False)
                # subset_array = np.append(subset_array, scores[0])
                for k in range(17):
                    results[count][k] = scores[0][k]
                count += 1
                # print(results)
        change = self.compute_change_between_levels(results)
        visualize_scores(results, change, 'Veränderung der Attribution Scores \n wenn sich beide Hyperparameter verändern', 'both_hyperparameters', 'Stufen', 0, 5, 1)


    def test_efficiency(
            self,
            **kwargs
    ): 
        
        test_indices = [281, 255, 686, 364, 197, 457, 171]
        for i in test_indices:
            x,y = self.data[i]
            y_res = self.model.predict(x)
            print("Instance: " + str(i))
            print("class prediction" + str(y_res))
            target_label_index = np.argmax(y_res.numpy())
            scores = self.explainer.attribute(x, target_label_index=target_label_index, subset_percentage=0.01, number_of_samples=50, kwargs=kwargs, log_odds=False)
            beta0 = scores[0][0]
            betasum = sum(scores[0][1:])
            print("scores: " + str(scores[0]))
            print("betasum: " + str(betasum))
            # print("beta0: " + str(beta0))
            print("betasum + beta0: " + str(sum(scores[0])))
            mean_prediction = self.get_mean_prediction(self.data, np.arange(len(self.data)))
            np.random.seed(43)
            indices = np.random.randint(len(self.explainer.data), size=80)
            mean_prediction_sampled = self.get_mean_prediction(self.explainer.data, indices)   
            print(mean_prediction_sampled[target_label_index])       
            # mean_prediction = self.model.predict(kwargs.get('baseline', torch.zeros(16)))
            # print("mean - beta0: " + str(mean_prediction[target_label_index]-beta0))
            # print("mean prediction: " + str(mean_prediction))
            efficiency_value = betasum - (max(y_res.numpy()) - mean_prediction[target_label_index])
            sampled_efficiency_value = betasum - (max(y_res.numpy()) - mean_prediction_sampled[target_label_index])
            print("Efficiency: " + str(efficiency_value))
            print("Efficiency of sampled average: " + str(sampled_efficiency_value))

    def compute_change_between_levels(
            self,
            results
    ): 
        levels = len(results)-1
        differences = [None]
        for i in range(len(results)-1):
            sum_of_differences = 0
            for j in range(17):
                sum_of_differences += abs(results[i+1][j]-results[i][j])
            differences.append(sum_of_differences)
        print(differences)
        return differences
            


    def get_mean_prediction(
        self,
        data,
        indices = None
    ):

        expectation = np.zeros(7)
        for index in indices:
            x, y = data[index]
            res = self.model.predict(x)
            expectation = np.add(expectation, res)
        mean_prediction = expectation / len(indices)
        # print(mean_prediction)
        return mean_prediction

            
        
                

    