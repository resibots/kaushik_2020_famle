
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import scipy.stats as stats
import numpy as np

class RS_opt(object):
    def __init__(self, config):
        self.max_iters = config["max_iters"]#20
        self.lb, self.ub = config["lb"], config["ub"]#-1, 1
        self.popsize = config["popsize"] #200
        self.sol_dim = config["sol_dim"] #2*10 #action dim*horizon
        self.cost_function = config["cost_fn"]

    def obtain_solution(self, init_mean=None, init_var=None):
        """Optimizes the cost function using the provided initial candidate distribution
        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        if init_mean is None or init_var is None:
            samples = np.random.uniform(self.lb, self.ub, size=(self.max_iters*self.popsize,self.sol_dim))
            costs = self.cost_function(samples)
            return samples[np.argmin(costs)]
        else:
            assert init_mean is not None and init_var is not None, "init mean and var must be provided"
            samples = np.random.normal(init_mean, init_var, size=(self.max_iters*self.popsize, self.sol_dim))
            samples = np.clip(samples, self.lb, self.ub)
            costs = self.cost_function(samples)
            return samples[np.argmin(costs)]

if __name__ == '__main__':
    from test_env import Point

    horizon = 10
    action_dim = 2
    goal = [-7, 14]
    env = Point(goal)
    env.reset()
    dummy_env = Point(goal)
    dummy_env.reset()

    def cost_fn(samples):
        global dummy_env
        current_state = env.state()
        costs = []
        for s in samples:
            dummy_env.reset(current_state)
            total_cost = 0
            for i in range(horizon):
                a = s[2*i:2*i+2]
                state, cost, _, _ = dummy_env.step(a)
                total_cost += cost

            costs.append(total_cost)
        return costs

    config = {
                "max_iters": 20, 
                "epsilon": 0.01, 
                "lb": -1, 
                "ub": 1,
                "popsize": 200,
                "sol_dim": action_dim*horizon, 
                "num_elites": 50,
                "cost_fn": cost_fn, 
                "alpha": 0.01
    }

    init_mean, init_var = np.zeros(config["sol_dim"]), np.ones(config["sol_dim"])* 0.5
    opt = RS_opt(config)
    for i in range(100):
        sol = opt.obtain_solution(init_mean, init_var)
        init_mean, init_var = np.zeros(config["sol_dim"]) , np.ones(config["sol_dim"])* 0.5 
        a = sol[0:2]
        _ , _, _, _ = env.step(a)
        env.render()