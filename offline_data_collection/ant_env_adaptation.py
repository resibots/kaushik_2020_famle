import fast_adaptation_embedding.env
from fast_adaptation_embedding.models.ffnn import FFNN_Ensemble_Model
from fast_adaptation_embedding.controllers.random_shooting import RS_opt
import torch
import numpy as np
import copy
import gym
import time
from datetime import datetime
import pickle
import os
from os import path
import argparse
from utils import ProgBar


class Cost(object):
    def __init__(self, ensemble_model, init_state, horizon, action_dim, goal, pred_high, pred_low):
        self.__ensemble_model = ensemble_model
        self.__init_state = init_state
        self.__horizon = horizon
        self.__action_dim = action_dim
        self.__goal = goal
        self.__models = self.__ensemble_model.get_models()
        self.__pred_high = pred_high
        self.__pred_low = pred_low
        self.__obs_dim = len(init_state)

    def cost_fn(self, samples):
        action_samples = torch.FloatTensor(samples).cuda(
        ) if self.__ensemble_model.CUDA else torch.FloatTensor(samples)
        init_states = torch.FloatTensor(np.repeat([self.__init_state], len(samples), axis=0)).cuda(
        ) if self.__ensemble_model.CUDA else torch.FloatTensor(np.repeat([self.__init_state], len(samples), axis=0))
        all_costs = torch.FloatTensor(np.zeros(len(samples))).cuda(
        ) if self.__ensemble_model.CUDA else torch.FloatTensor(np.zeros(len(samples)))

        n_batch = max(1, int(len(samples)/1024))
        per_batch = len(samples)/n_batch

        for i in range(n_batch):
            start_index = int(i*per_batch)
            end_index = len(samples) if i == n_batch - \
                1 else int(i*per_batch + per_batch)
            action_batch = action_samples[start_index:end_index]
            start_states = init_states[start_index:end_index]
            dyn_model = self.__models[np.random.randint(0, len(self.__models))]
            for h in range(self.__horizon):
                actions = action_batch[:, h*self.__action_dim: h *
                                       self.__action_dim + self.__action_dim]
                model_input = torch.cat((start_states, actions), dim=1)
                diff_state = dyn_model.predict_tensor(model_input)
                start_states += diff_state
                for dim in range(self.__obs_dim):
                    start_states[:, dim].clamp_(
                        self.__pred_low[dim], self.__pred_high[dim])

                action_cost = torch.sum(actions * actions, dim=1) * 0.0
                x_vel_cost = -start_states[:, 13]
                survive_cost = (start_states[:, 0] < 0.26).type(
                    start_states.dtype) * 2.0
                all_costs[start_index: end_index] += x_vel_cost * config["discount"]**h + \
                    action_cost * config["discount"]**h + \
                    survive_cost * config["discount"]**h
        return all_costs.cpu().detach().numpy()


def train_ensemble_model(train_in, train_out, sampling_size, config, model=None):
    network = model
    if network is None:
        network = FFNN_Ensemble_Model(dim_in=config["ensemble_dim_in"],
                                      hidden=config["ensemble_hidden"],
                                      hidden_activation=config["hidden_activation"],
                                      dim_out=config["ensemble_dim_out"],
                                      CUDA=config["ensemble_cuda"],
                                      SEED=config["ensemble_seed"],
                                      output_limit=config["ensemble_output_limit"],
                                      dropout=config["ensemble_dropout"],
                                      n_ensembles=config["n_ensembles"])
    network.train(epochs=config["ensemble_epoch"], training_inputs=train_in, training_targets=train_out,
                  batch_size=config["ensemble_batch_size"], logInterval=config["ensemble_log_interval"], sampling_size=sampling_size)
    return copy.deepcopy(network)


def process_data(data):
    '''Assuming dada: an array containing [state, action, state_transition, cost] '''
    training_in = []
    training_out = []
    for d in data:
        s = d[0]
        a = d[1]
        training_in.append(np.concatenate((s, a)))
        training_out.append(d[2])
    return np.array(training_in), np.array(training_out), np.max(training_in, axis=0), np.min(training_in, axis=0)


def execute_random(env, steps, init_state):
    current_state = env.reset()
    trajectory = []
    traject_cost = 0
    for i in range(steps):
        a = env.action_space.sample()
        next_state, r = 0, 0
        for k in range(1):
            next_state, r, _, _ = env.step(a)

        trajectory.append([current_state.copy(), a.copy(),
                          next_state-current_state, -r])
        current_state = next_state
        traject_cost += -r
    return np.array(trajectory), traject_cost


def execute(env, init_state, steps, init_mean, init_var, model, config, last_action_seq, pred_high, pred_low):
    current_state = env.reset()
    trajectory = []
    traject_cost = 0
    model_error = 0
    sliding_mean = np.zeros(config["sol_dim"])
    random = np.random.rand()
    mutation = np.random.rand(config["sol_dim"]) * 2. * 0.5 - 0.5
    rand = np.random.rand(config["sol_dim"])
    mutation *= np.array([1.0 if r > 0.25 else 0.0 for r in rand])
    bar = ProgBar(steps, track_time=True,
                  title='\nExecuting....', bar_char='▒')
    for i in range(steps):
        cost_object = Cost(ensemble_model=model, init_state=current_state,
                           horizon=config["horizon"], action_dim=env.action_space.shape[0], goal=config["goal"], pred_high=pred_high, pred_low=pred_low)
        config["cost_fn"] = cost_object.cost_fn
        optimizer = RS_opt(config)
        sol = optimizer.obtain_solution()
        # Take soft action
        # if i == 0 else sol[0:env.action_space.shape[0]] * 0.8 + a * 0.2
        a = sol[0:env.action_space.shape[0]]
        next_state, r = 0, 0
        for k in range(1):
            next_state, r, _, _ = env.step(a)
        trajectory.append([current_state.copy(), a.copy(),
                          next_state-current_state, -r])
        model_error += test_model(model, current_state.copy(),
                                  a.copy(), next_state-current_state)
        current_state = next_state
        traject_cost += -r
        sliding_mean[0:-len(a)] = sol[len(a)::]
        bar.update(item_id=" Step " + str(i) + " ")

    print("Model error: ", model_error)
    return np.array(trajectory), traject_cost


def test_model(ensemble_model, init_state, action, state_diff):
    x = np.concatenate(([init_state], [action]), axis=1)
    y = state_diff.reshape(1, -1)
    y_pred = ensemble_model.get_models()[0].predict(x)
    # print("True: ", y.flatten())
    # print("pred: ", y_pred.flatten())
    # input()
    return np.power(y-y_pred, 2).sum()


def extract_action_seq(data):
    actions = []
    for d in data:
        actions += d[1].tolist()
    return np.array(actions)


def main(gym_args, mismatches, config, gym_kwargs={}):
    '''---------Prepare the directories------------------'''
    now = datetime.now()
    timestamp = now.strftime("%d_%m_%Y_%H_%M_%S")
    experiment_name = timestamp + "_" + config["exp_suffix"]
    res_dir = os.path.join(
        os.getcwd(), config["result_dir"], config["env_name"], experiment_name)
    try:
        i = 0
        while True:
            res_dir += "_" + str(i)
            i += 1
            if not os.path.isdir(res_dir):
                os.makedirs(res_dir)
                os.makedirs(res_dir+"/videos")
                break
    except:
        print("Could not make the result directory!!!")

    with open(res_dir + "/details.txt", "w+") as f:
        f.write(config["exp_details"])

    with open(res_dir + '/config.json', 'w') as fp:
        import json
        json.dump(config, fp)

    # **********************************
    n_task = len(mismatches)
    data = n_task * [None]
    models = n_task * [None]
    best_action_seq = np.random.rand(config["sol_dim"])*2.0 - 1.0
    best_cost = 10000
    last_action_seq = None
    all_action_seq = []
    all_costs = []

    '''-------------Attempt to load saved data------------------'''
    if path.exists(config["data_dir"] + "/trajectories.npy") and path.exists(config["data_dir"] + "/mismatches.npy"):
        print("Found stored data. Setting random trials to zero.")
        data = np.load(config["data_dir"] + "/trajectories.npy")
        mismatches = np.load(config["data_dir"] + "/mismatches.npy")
        config["random_episodes"] = 0
        n_task = len(mismatches)

    envs = [gym.make(*gym_args, **gym_kwargs) for i in range(n_task)]
    for i, e in enumerate(envs):
        e.set_mismatch(mismatches[i])

    for i in range(n_task):
        with open(res_dir + "/costs_task_" + str(i)+".txt", "w+") as f:
            f.write("")

    np.save(res_dir + '/mismatches.npy', mismatches)
    for index_iter in range(config["iterations"]*n_task):
        '''Pick a random environment'''
        env_index = int(index_iter % n_task)  # np.random.randint(0, n_task)
        env = envs[env_index]

        print("Episode: ", index_iter)
        print("Env index: ", env_index)
        c = None

        if data[env_index] is None or index_iter < config["random_episodes"]*n_task:
            print("Execution (Random actions)...")
            trajectory, c = execute_random(
                env=env, steps=config["episode_length"], init_state=config["init_state"])
            if data[env_index] is None:
                data[env_index] = trajectory
            else:
                data[env_index] = np.concatenate(
                    (data[env_index], trajectory), axis=0)
            print("Cost : ", c)

            if c < best_cost:
                best_cost = c
                best_action_seq = []
                for d in trajectory:
                    best_action_seq += d[1].tolist()
                best_action_seq = np.array(best_action_seq)
                last_action_seq = best_action_seq
            all_action_seq.append(extract_action_seq(trajectory))
            all_costs.append(c)
        else:
            '''------------Update models------------'''
            x, y, high, low = process_data(data[env_index])
            # print(high)
            # print(low)
            # input()
            print("Learning model...")
            models[env_index] = train_ensemble_model(
                train_in=x, train_out=y, sampling_size=-1, config=config, model=models[env_index])
            print("Execution...")

            trajectory, c = execute(env=env,
                                    init_state=config["init_state"],
                                    model=models[env_index],
                                    steps=config["episode_length"],
                                    init_mean=best_action_seq[0:config["sol_dim"]],
                                    init_var=0.1 * np.ones(config["sol_dim"]),
                                    config=config,
                                    last_action_seq=best_action_seq,
                                    pred_high=high,
                                    pred_low=low)
            data[env_index] = np.concatenate(
                (data[env_index], trajectory), axis=0)
            print("Cost : ", c)

            if c < best_cost:
                best_cost = c
                best_action_seq = []
                for d in trajectory:
                    best_action_seq += d[1].tolist()
                best_action_seq = np.array(best_action_seq)
                last_action_seq = extract_action_seq(trajectory)

            all_action_seq.append(extract_action_seq(trajectory))
            all_costs.append(c)

        if index_iter % config["dump_trajects"] == 0 and index_iter > 0:
            print("Saving trajectories..")
            np.save(res_dir + "/trajectories.npy", data)
        with open(res_dir + "/costs_task_" + str(env_index)+".txt", "a+") as f:
            f.write(str(c)+"\n")

        print("-------------------------------\n")

    print("Finally Saving trajectories..")
    np.save(res_dir + "/trajectories.npy", data)

################################################################################


config = {
    # exp parameters:
    "horizon": 20,  # NOTE: "sol_dim" must be adjusted
    "iterations": 120,
    "random_episodes": 100,  # per task
    "episode_length": 1000,
    "init_state": None,  # Must be updated before passing config as param
    "action_dim": 8,
    "goal": None,  # Not used here

    # logging
    "record_video": False,
    "result_dir": "results",
    "env_name": "ensemble_ant",
    "exp_suffix": "experiment",
    "exp_details": "Ant default",
    "dump_trajects": 100,
    "data_dir": "data/ant_data",


    # Ensemble model params
    "cuda": True,
    "ensemble_epoch": 5,
    "ensemble_dim_in": 8+27,
    "ensemble_dim_out": 27,
    "ensemble_hidden": [200, 200, 100],
    "hidden_activation": "relu",
    "ensemble_cuda": True,
    "ensemble_seed": None,
    "ensemble_output_limit": None,
    "ensemble_dropout": 0.0,
    "n_ensembles": 1,
    "ensemble_batch_size": 64,
    "ensemble_log_interval": 500,

    # Optimizer parameters
    "max_iters": 5,
    "epsilon": 0.0001,
    "lb": -1.,
    "ub": 1.,
    "popsize": 2000,
    "sol_dim": 8*20,  # NOTE: Depends on Horizon
    "num_elites": 50,
    "cost_fn": None,
    "alpha": 0.1,
    "discount": 1.
}

# optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("--iterations",
                    help='Total episodes.',
                    type=int)
parser.add_argument("--episode_length",
                    help='Total time steps in Episodes.',
                    type=int)
parser.add_argument("--random_episodes",
                    help='Random Episodes.',
                    type=int)
parser.add_argument("--exp_details",
                    help='Details about the experiment',
                    type=str)
parser.add_argument("--data_dir",
                    help='To load trajectories from',
                    type=str)
parser.add_argument("--dump_trajects",
                    help='Create trajectory.npy after every dump_trajects iterations',
                    type=int)
arguments = parser.parse_args()
if arguments.iterations is not None:
    config['iterations'] = arguments.iterations
if arguments.episode_length is not None:
    config['episode_length'] = arguments.episode_length
if arguments.random_episodes is not None:
    config['random_episodes'] = arguments.random_episodes
if arguments.exp_details is not None:
    config['exp_details'] = arguments.exp_details
if arguments.dump_trajects is not None:
    config['dump_trajects'] = arguments.dump_trajects
if arguments.data_dir is not None:
    config['data_dir'] = arguments.data_dir

ang_diff = 30./360.
mismatches = np.array([[1., 1., 1., 1., 1., 1., 1., 1., 0.],
                       [0., 1., 1., 1., 1., 1., 1., 1., 0.],
                       [1., 0., 1., 1., 1., 1., 1., 1., 0.],
                       [1., 1., 0., 1., 1., 1., 1., 1., 0.],
                       [1., 1., 1., 0., 1., 1., 1., 1., 0.],
                       [1., 1., 1., 1., 0., 1., 1., 1., 0.],
                       [1., 1., 1., 1., 1., 0., 1., 1., 0.],
                       [1., 1., 1., 1., 1., 1., 0., 1., 0.],
                       [1., 1., 1., 1., 1., 1., 1., 0., 0.],
                       [1., 1., 1., 1., 1., 1., 1., 1., ang_diff],
                       [1., 1., 1., 1., 1., 1., 1., 1., 2*ang_diff],
                       [1., 1., 1., 1., 1., 1., 1., 1., 3*ang_diff],
                       [1., 1., 1., 1., 1., 1., 1., 1., 4*ang_diff],
                       [1., 1., 1., 1., 1., 1., 1., 1., 5*ang_diff],
                       [1., 1., 1., 1., 1., 1., 1., 1., 6*ang_diff],
                       [1., 1., 1., 1., 1., 1., 1., 1., 7*ang_diff],
                       [1., 1., 1., 1., 1., 1., 1., 1., 8*ang_diff],
                       [1., 1., 1., 1., 1., 1., 1., 1., 9*ang_diff],
                       [1., 1., 1., 1., 1., 1., 1., 1., 10*ang_diff],
                       [1., 1., 1., 1., 1., 1., 1., 1., 11*ang_diff]])

# envs = [gym.make("AntMuJoCoEnv_fastAdapt-v0") for i in range(len(mismatches))]
gym_args = ["AntMuJoCoEnv_fastAdapt-v0"]
main(gym_args=gym_args, mismatches=mismatches, config=config)
