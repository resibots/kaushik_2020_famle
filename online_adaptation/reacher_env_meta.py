import fast_adaptation_embedding.env
import fast_adaptation_embedding.models.famle as nn_model
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
# from pyprind import ProgBar
from utils import ProgBar
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import argparse


class Cost(object):
    def __init__(self, model, init_state, horizon, action_dim, goal, task_likelihoods, pred_high, pred_low):
        self.__models = model
        self.__goal = torch.FloatTensor([goal]).cuda(
        ) if self.__models[0].cuda_enabled else torch.FloatTensor([goal])
        self.__init_state = init_state
        self.__horizon = horizon
        self.__action_dim = action_dim
        self.__task_likelihoods = np.array(task_likelihoods)
        self.__n_tasks = len(task_likelihoods)
        self.__pred_high = pred_high
        self.__pred_low = pred_low
        self.__obs_dim = len(init_state)

    def cost_fn(self, samples):
        action_samples = torch.FloatTensor(samples).cuda(
        ) if self.__models[0].cuda_enabled else torch.FloatTensor(samples)
        init_states = torch.FloatTensor(np.repeat([self.__init_state], len(samples), axis=0)).cuda(
        ) if self.__models[0].cuda_enabled else torch.FloatTensor(np.repeat([self.__init_state], len(samples), axis=0))
        all_costs = torch.FloatTensor(np.zeros(len(samples))).cuda(
        ) if self.__models[0].cuda_enabled else torch.FloatTensor(np.zeros(len(samples)))

        n_batch = max(1, int(len(samples)/1024))
        per_batch = len(samples)/n_batch

        for i in range(n_batch):
            start_index = int(i*per_batch)
            end_index = len(samples) if i == n_batch - \
                1 else int(i*per_batch + per_batch)
            action_batch = action_samples[start_index:end_index]
            start_states = init_states[start_index:end_index]
            dyn_model = self.__models[np.argmax(self.__task_likelihoods)]
            for h in range(self.__horizon):
                actions = action_batch[:, h*self.__action_dim: h *
                                       self.__action_dim + self.__action_dim]
                model_input = torch.cat((start_states, actions), dim=1)
                diff_state = dyn_model.predict_tensor(model_input)
                start_states += diff_state

                for dim in range(self.__obs_dim):
                    start_states[:, dim].clamp_(
                        self.__pred_low[dim], self.__pred_high[dim])

                eff_cost = - \
                    torch.exp(-torch.pow(start_states[:, -
                              2::] - self.__goal, 2).sum(dim=1))
                all_costs[start_index: end_index] += eff_cost * \
                    config["discount"]**h
        return all_costs.cpu().detach().numpy()


def train_meta(tasks_in, tasks_out, config):
    model = nn_model.Embedding_NN(dim_in=config["dim_in"],
                                  hidden=config["hidden_layers"],
                                  dim_out=config["dim_out"],
                                  embedding_dim=config["embedding_size"],
                                  num_tasks=len(tasks_in),
                                  CUDA=config["cuda"],
                                  SEED=None,
                                  output_limit=config["output_limit"],
                                  dropout=0.0,
                                  hidden_activation=config["hidden_activation"])
    nn_model.train_meta(model,
                        tasks_in,
                        tasks_out,
                        meta_iter=config["meta_iter"],
                        inner_iter=config["inner_iter"],
                        inner_step=config["inner_step"],
                        meta_step=config["meta_step"],
                        minibatch=config["meta_batch_size"],
                        inner_sample_size=config["inner_sample_size"])
    return model


def train_model(model, train_in, train_out, task_id, config):
    cloned_model = copy.deepcopy(model)
    nn_model.train(cloned_model,
                   train_in,
                   train_out,
                   task_id=task_id,
                   inner_iter=config["epoch"],
                   inner_lr=config["learning_rate"],
                   minibatch=config["minibatch_size"])
    return cloned_model


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
        # env.joint_reset()
        trajectory.append([current_state.copy(), a.copy(),
                          next_state-current_state, -r])
        current_state = next_state
        traject_cost += -r

    return np.array(trajectory), traject_cost


def execute(env, init_state, steps, init_mean, init_var, model, config, last_action_seq, task_likelihoods, pred_high, pred_low, recorder):
    # current_state = env.reset()
    current_state = copy.copy(env.state) if config['online'] else env.reset()
    try:
        config["goal"] = env.goal
    except:
        pass
    trajectory = []
    traject_cost = 0
    sliding_mean = init_mean  # np.zeros(config["sol_dim"])

    temp_config = copy.deepcopy(config)
    temp_config["popsize"] = 20000
    optimizer = None
    sol = None
    bar = ProgBar(steps, track_time=True,
                  title='\nExecuting....', bar_char='▒')
    for i in range(steps):
        cost_object = Cost(model=model, init_state=current_state, horizon=config["horizon"], task_likelihoods=task_likelihoods,
                           action_dim=env.action_space.shape[0], goal=config["goal"], pred_high=pred_high, pred_low=pred_low)
        config["cost_fn"] = cost_object.cost_fn
        optimizer = RS_opt(config)
        # sol = optimizer.obtain_solution(sliding_mean, init_var)
        sol = optimizer.obtain_solution()

        a = sol[0:env.action_space.shape[0]]
        next_state, r = 0, 0
        for k in range(1):
            if config["record_video"]:
                recorder.capture_frame()
            next_state, r, _, _ = env.step(a)

        # env.joint_reset()
        trajectory.append([current_state.copy(), a.copy(),
                          next_state-current_state, -r])
        current_state = next_state
        traject_cost += -r

        # sliding_mean = last_action_seq[i*config["sol_dim"] : (i+1) * config["sol_dim"]]
        # sliding_mean[0:-len(a)] = sol[len(a)::]
        # sliding_mean[-len(a)::] = sol[-len(a)::]
        bar.update(item_id=" Step " + str(i) + " ")

    if config["record_video"]:
        recorder.capture_frame()
        recorder.close()
    return trajectory, traject_cost


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


def compute_likelihood(data, models, adapt_steps, beta=1.0):
    '''
    Computes MSE loss and then softmax to have a probability
    '''
    data_size = config['adapt_steps']
    if data_size is None:
        data_size = len(data)
    lik = np.zeros(len(models))
    x, y, _, _ = process_data(data[-data_size::])
    for i, m in enumerate(models):
        y_pred = m.predict(x)
        lik[i] = np.exp(- beta * m.loss_function_numpy(y, y_pred)/len(x))
    return lik/np.sum(lik)


def sample_model_index(likelihoods):
    cum_sum = np.cumsum(likelihoods)
    num = np.random.rand()
    for i, cum_prob in enumerate(cum_sum):
        if num <= cum_prob:
            return i


def main(gym_args, config, mismatch_fn, gym_kwargs={}):
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

    '''---------Prepare the test environment---------------'''
    env = gym.make(*gym_args, **gym_kwargs)
    trained_mismatches = np.load(config["data_dir"] + "/mismatches.npy")
    n_training_tasks = len(trained_mismatches)
    try:
        s = os.environ['DISPLAY']
        print("Display available")
        # env.render(mode="rgb_array")
        env.render(mode="human")
        env.reset()
    except:
        print("Display not available")
        env.reset()

    print("\n\n\n")
    '''---------Initialize global variables------------------'''
    data = []
    models = []
    best_action_seq = np.random.rand(config["sol_dim"])*2.0 - 1.0
    best_cost = 10000
    last_action_seq = None
    all_action_seq = []
    all_costs = []
    with open(res_dir + "/costs.txt", "w+") as f:
        f.write("")

    '''--------------------Meta learn the models---------------------------'''
    meta_model = None
    if not path.exists(config["data_dir"] + "/" + config["model_name"]+".pt"):
        print("Model not found. Learning from data...")
        meta_data = np.load(config["data_dir"] +
                            "/trajectories.npy", allow_pickle=True)
        tasks_in, tasks_out = [], []
        for n in range(n_training_tasks):
            x, y, high, low = process_data(meta_data[n])
            tasks_in.append(x)
            tasks_out.append(y)
            print("task ", n, " data: ", len(tasks_in[n]), len(tasks_out[n]))
        meta_model = train_meta(tasks_in, tasks_out, config)
        meta_model.save(config["data_dir"] + "/" + config["model_name"]+".pt")
    else:
        print("Model found. Loading from '.pt' file...")
        device = torch.device(
            "cuda") if config["cuda"] else torch.device("cpu")
        meta_model = nn_model.load_model(
            config["data_dir"] + "/" + config["model_name"]+".pt", device)

    raw_models = [copy.deepcopy(meta_model) for _ in range(n_training_tasks)]
    models = [copy.deepcopy(meta_model) for _ in range(n_training_tasks)]
    for task_id, m in enumerate(raw_models):
        m.fix_task(task_id)

    for task_id, m in enumerate(models):
        m.fix_task(task_id)

    '''------------------------Test time------------------------------------'''

    high, low = np.ones(config["dim_out"])*1000.,  - \
        np.ones(config["dim_out"])*1000.
    task_likelihoods = np.random.rand(n_training_tasks)

    for index_iter in range(config["iterations"]):
        print("Episode: ", index_iter)
        new_mismatch = mismatch_fn(config)
        print("Mismatch: ", new_mismatch.tolist())
        env.set_mismatch(new_mismatch)
        recorder = VideoRecorder(
            env, res_dir + "/videos/" + str(index_iter) + ".mp4") if config["record_video"] else None
        trajectory, c = execute(env=env,
                                init_state=config["init_state"],
                                model=models,
                                steps=config["episode_length"],
                                init_mean=np.zeros(config["sol_dim"]),
                                init_var=0.01 * np.ones(config["sol_dim"]),
                                config=config,
                                last_action_seq=None,
                                task_likelihoods=task_likelihoods,
                                pred_high=high,
                                pred_low=low,
                                recorder=recorder)

        data += trajectory
        '''-----------------Compute likelihood before relearning the models-------'''
        task_likelihoods = compute_likelihood(
            data, raw_models, config['adapt_steps'])
        print("\nlikelihoods: ", task_likelihoods)

        x, y, high, low = process_data(data)

        task_index = sample_model_index(
            task_likelihoods) if config["sample_model"] else np.argmax(task_likelihoods)
        print("\nEstimated task-id: ", task_index)
        task_likelihoods = task_likelihoods * 0
        task_likelihoods[task_index] = 1.0
        data_size = config['adapt_steps']
        if data_size is None:
            data_size = len(x)
        print("Learning model with recent ", data_size, " data")
        models[task_index] = train_model(model=copy.deepcopy(
            raw_models[task_index]), train_in=x[-data_size::], train_out=y[-data_size::], task_id=task_index, config=config)

        print("\nCost : ", c)
        with open(res_dir + "/costs.txt", "a+") as f:
            f.write(str(c)+"\n")

        if c < best_cost:
            best_cost = c
            best_action_seq = []
            for d in trajectory:
                best_action_seq += d[1].tolist()
            best_action_seq = np.array(best_action_seq)
            last_action_seq = extract_action_seq(trajectory)

        all_action_seq.append(extract_action_seq(trajectory))
        all_costs.append(c)

        np.save(res_dir + "/trajectories.npy", data)
        print("\n********************************************************\n")

#######################################################################################################


config = {
    # exp parameters:
    "horizon": 10,  # NOTE: "sol_dim" must be adjusted
    "iterations": 100,
    "episode_length": 50,
    "online": False,
    "adapt_steps": None,
    "init_state": None,  # Must be updated before passing config as param
    "action_dim": 8,
    "goal": [0, 0],  # NOTE: Note used here.
    "record_video": False,
    "online_damage_probability": 0.0,
    "sample_model": False,

    # logging
    "result_dir": "results",
    "data_dir": "data/reacher_data",
    "model_name": "reacher_meta_embedding_model",
    "env_name": "meta_reacher",
    "exp_suffix": "experiment",
    "exp_details": "Default experiment.",

    # Model_parameters
    "dim_in": 5+12,
    "dim_out": 12,
    "hidden_layers": [70, 50],
    "embedding_size": 8,
    "cuda": True,
    "output_limit": 10.0,

    # Meta learning parameters
    "meta_iter": 5000,  # Total meta update steps
    "meta_step": 0.3,
    "inner_iter": 10,  # Innner optimization steps
    "inner_step": 0.0001,
    "meta_batch_size": 32,
    "inner_sample_size": 500,

    # Model learning parameters
    "epoch": 20,
    "learning_rate": 1e-4,
    "minibatch_size": 32,
    "hidden_activation": "relu",

    # Optimizer parameters
    "max_iters": 5,
    "epsilon": 0.0001,
    "lb": -1.,
    "ub": 1.,
    "popsize": 2000,
    "sol_dim": 5*10,  # NOTE: Depends on Horizon
    "num_elites": 30,
    "cost_fn": None,
    "alpha": 0.1,
    "discount": 1.0
}

# optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("--iterations",
                    help='Total episodes in episodic learning. Total MPC steps in the experiment.',
                    type=int)
parser.add_argument("--data_dir",
                    help='Path to load dynamics data and/or model',
                    type=str)
parser.add_argument("--exp_details",
                    help='Details about the experiment',
                    type=str)
parser.add_argument("--online",
                    action='store_true',
                    help='Will not reset back to init position', )
parser.add_argument("--adapt_steps",
                    help='Past steps to be used to learn a new model from the meta model',
                    type=int)
parser.add_argument("--control_steps",
                    help='Steps after which learn a new model => Learning frequency.',
                    type=int)
parser.add_argument("--rand_motor_damage",
                    action='store_true',
                    help='Sample a random joint damage.')
parser.add_argument("--rand_orientation_fault",
                    action='store_true',
                    help='Sample a random orientation estimation fault.')
parser.add_argument("--sample_model",
                    action='store_true',
                    help='Sample a model (task-id) using the likelihood information. Default: Picks the most likely model.')
parser.add_argument("--online_damage_probability",
                    help='Sample probabilistically random mismatch during mission. NOT used for episodic testing',
                    default=0.0,
                    type=float)

arguments = parser.parse_args()
if arguments.data_dir is not None:
    config['data_dir'] = arguments.data_dir
if arguments.iterations is not None:
    config['iterations'] = arguments.iterations
if arguments.exp_details is not None:
    config['exp_details'] = arguments.exp_details
if arguments.online is True:
    config['online'] = True
    if arguments.adapt_steps is not None:
        config['adapt_steps'] = arguments.adapt_steps
    if arguments.control_steps is not None:
        config['episode_length'] = arguments.control_steps
    if arguments.online_damage_probability is not None:
        config['online_damage_probability'] = arguments.online_damage_probability
    print("Online learning with adaptation steps: ",
          config['adapt_steps'], " control steps: ", config['episode_length'])
else:
    print("Episodic learning with episode length: ", config['episode_length'])

if arguments.rand_motor_damage is not None:
    config['rand_motor_damage'] = arguments.rand_motor_damage
if arguments.rand_orientation_fault is not None:
    config['rand_orientation_fault'] = arguments.rand_orientation_fault
if arguments.sample_model is not None:
    config['sample_model'] = arguments.sample_model

'''----------- Environment specific setup --------------'''


def sample_mismatch(conf):
    '''
    If beginning of the experiment, then samples a damage.
    Else sample according to the given probability.
    '''
    if np.random.rand() < config['online_damage_probability'] or not conf.get("mismatch"):
        mismatches = np.array([-1., 1., 1., 0., 1., 0., 0., 0., 0., 0.])
        if conf['rand_motor_damage']:
            print(
                "WARNING: rand_motor_damage fault NOT used. Fixed damage for better comparision.")
        if conf['rand_orientation_fault'] is True:
            print("WARNING: Orientation fault NOT used")
        conf["mismatch"] = mismatches.tolist()
    return np.array(conf["mismatch"])


args = ["Reacher-v0"]
kwargs = {"goal_sampling": False}
main(gym_args=args, gym_kwargs=kwargs,
     config=config, mismatch_fn=sample_mismatch)
