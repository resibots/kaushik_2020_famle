from fast_adaptation_embedding.models import famle
import numpy as np
import matplotlib.pyplot as plt


def random_task(num):
    phase = np.random.uniform(low=-0.8*np.pi, high=0.8*np.pi)
    ampl = np.random.uniform(0.3, 1.0)

    if num % 2 == 0.0:
        def f_randomsine(x): return np.sin(x + phase) * ampl
    else:
        def f_randomsine(x): return np.sin(x + phase) * ampl
    return f_randomsine


def generate_data(num_tasks):
    d_in = []
    d_out = []
    for i in range(num_tasks):
        x = np.linspace(-6, 6, np.random.randint(90, 100)).reshape(-1, 1)
        f = random_task(i)
        y = f(x)
        d_in.append(x)
        d_out.append(y)
    return d_in, d_out

# Parameters


# Number of tasks to meta-train train the model
num_tasks = 10

# Total meta gradient steps
meta_iter = 1000

# Initial size of meta-gradient step. It is decreases over time with linear schedule
m_step = 0.1

# Inner optimization steps
inner_iter = 100

# Inner gradient step
n_step = 0.001

# Number of traing samples (After meta-training)
n_training = 3


# Generate data for meta training
tasks_in, tasks_out = generate_data(num_tasks)

# Generate data for the test task
x, y = generate_data(1)
test_in, test_out = x[0], y[0]

# Sample a few training data
indices = np.random.permutation(len(test_in))[0:n_training]
train_in, train_out = test_in[indices], test_out[indices]

# Initialize the model
# This is a neural netwrok that takes both the data as well as the embedding of the task id as input to predict the output
model = famle.Embedding_NN(dim_in=1, hidden=[20, 20, 20], dim_out=1, embedding_dim=5,
                           num_tasks=len(tasks_in), CUDA=True, SEED=None, output_limit=None, dropout=0.0)

# Meta train the model + embedding, and save it
famle.train_meta(model, tasks_in, tasks_out, meta_iter=meta_iter, inner_iter=inner_iter, inner_step=n_step, meta_step=m_step, minibatch=128)
model.save("model.pt")

# If already trained model available, just load it
model = famle.load_model("model.pt")

# Estimate the most likely task-id for from the real data
most_likely_id = famle.most_likely_taskid(train_in, train_out, model)

# Set the most likely task id in the model, i.e fix the task-embedding input to the model
model.fix_task(most_likely_id)

# Check the model before fitting the trainig data
predict_before = model.predict(test_in)
plt.plot(test_in, predict_before, '--b', alpha=0.5,
         label="Before adaptation (meta-trained only)")

# Now train the model with the data
famle.train(model, train_in, train_out, task_id=most_likely_id,
            inner_iter=1000, inner_lr=1e-3, minibatch=32)

# Check the trained model
predict_after = model.predict(test_in)
plt.plot(test_in, predict_after, '-b', label="After adaptation")
plt.plot(test_in, test_out, '-r', label="Real task", alpha=0.6)
plt.plot(train_in, train_out, 'ok', markersize=8, alpha=0.5)
for i in range(len(tasks_in)):
    plt.plot(tasks_in[i], tasks_out[i], '-g', alpha=0.1)
plt.plot([0], [0], '-g', alpha=0.1, label="Meta training tasks")

plt.legend()
plt.show()
