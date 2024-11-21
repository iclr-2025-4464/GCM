import numpy as np
from ucimlrepo import fetch_ucirepo
from model import *

_SEED = 0

tf.random.set_seed(_SEED)
adult = fetch_ucirepo(id=2)
X = adult.data.features
y = adult.data.targets

xs, rs = [], []
num = 48842
for n in ['education-num', 'capital-gain', 'age', 'hours-per-week']:
    tmp = X[n].tolist()
    a, b = min(tmp), max(tmp)
    tmp = [(i-a)/(b-a) for i in tmp]
    rs.append(tmp)
for n in ['workclass', 'education', 'race', 'sex']:
    tmp = X[n].tolist()
    k = set(tmp)
    k = list(k)
    l = len(k)
    x = [k.index(i) for i in tmp]
    xx = [[0.0] * l for i in x]
    for i, j in zip(x, xx):
        j[i] = 1.0
    xs.append(xx)

r = [list(i) for i in zip(*rs)]
x = [sum(i, []) for i in zip(*xs)]
y = [1.0 if i == '>50K.' else 0 for i in y['income'].tolist()]

train_num = int(num * 0.8)
rnd_id = list(range(num))
np.random.seed(_SEED)
np.random.shuffle(rnd_id)
x_train = np.float32(np.array([x[i] for i in rnd_id[:train_num]]))
r_train = np.float32(np.array([r[i] for i in rnd_id[:train_num]]))
y_train = np.float32(np.array([y[i] for i in rnd_id[:train_num]]))
x_test = np.float32(np.array([x[i] for i in rnd_id[train_num:]]))
r_test = np.float32(np.array([r[i] for i in rnd_id[train_num:]]))
y_test = np.float32(np.array([y[i] for i in rnd_id[train_num:]]))


def train_step(x, r, y, model, lr = 0.001, reg_ratio = 1e-6):
    vars = model.variables
    with tf.GradientTape() as t:
        l2_reg = sum([tf.reduce_sum(tf.square(var)) for var in vars])
        y_pred, loss = model([x, r], y=y)
        total_loss = loss + l2_reg * reg_ratio
    if lr > 0:
        grads = t.gradient(total_loss, vars)
        grads = [tf.convert_to_tensor(i) for i in grads]
        for grad, var in zip(grads, vars):
            var.assign_sub(lr * tf.clip_by_value(grad, -100, 100))
    if lr==0:
        pass
    return y_pred, loss

def get_auc(y, y_pred, pos_ratio):
    tmp = sorted(zip(y_pred, y), key=lambda x: x[0])
    n, m, l, t = 0, 0, 0, 0
    d = len(y) * (1-pos_ratio)
    for j, (r, k) in enumerate(tmp):
        if k == 0:
            n += 1
        else:
            m += n
        if j < d:
            if k == 0:
                l += 1
        else:
            if k == 1:
                l += 1
        t += (r - k) ** 2
    auc = m / (n * (len(y) - n))
    acc = l / len(y)
    rms = (t / len(y)) ** 0.5
    return auc, acc, rms


model_list = [
    MlpModel(name=f'MLP_{_SEED}'),
    MinmaxModel(name=f'MM_{_SEED}'), # **
    SmoothedMinmaxModel(name=f'SMM_{_SEED}'), # **
    ConstrainedModel(name=f'CMNN_{_SEED}'), # **
    LatticeModel(name=f'Lattice_{_SEED}', k_list=[2,10,2,2]),
    HintModel(name=f'Hint_{_SEED}'), # **
    PwlModel(name=f'PWL_{_SEED}'),
    GenCostModel(name=f'GCM_{_SEED}', sample_num=64, kl_ratio=0.3),
    GcmViModel(name=f'GCM_VI_{_SEED}', sample_num=64, kl_ratio=0.3, elb_ratio=0.5),
    GcmViRevModel(name=f'GCM_VI_R_{_SEED}', sample_num=64, kl_ratio=0.3, elb_ratio=0.5),
]

train_loss_history_list = [[] for _ in model_list]
train_auc_history_list = [[] for _ in model_list]
train_acc_history_list = [[] for _ in model_list]
train_rms_history_list = [[] for _ in model_list]
report_interval = 100
step = 0
N = 100
total_step = N * report_interval
y_pred = tf.zeros_like(y_train)
lr = 0.3
mini_batch_size = 256
pos_ratio = sum(y_train) / len(y_train)
while step <= total_step:
    do_report = step % report_interval == 0
    ids = tf.random.categorical(tf.zeros_like([y_train]), mini_batch_size)  # [1, b]
    ids = tf.reshape(ids, [-1])
    x = tf.gather(x_train, ids, 0)
    r = tf.gather(r_train, ids, 0)
    y = tf.gather(y_train, ids, 0)
    if not do_report:
        for model in model_list:
            _, _ = train_step(x, r, y, model, lr)
        lr *= 0.9997
    else:
        info = f'step: {step} lr: {lr:4f}\nmodel     \tloss    \trms     \tauc     \tacc     \n'
        for model, loss_history, auc_history, acc_history, rms_history in (
                zip(model_list, train_loss_history_list, train_auc_history_list, train_acc_history_list,
                    train_rms_history_list)):
            y_pred, loss = train_step(x_test, r_test, y_test, model, 0)
            loss_history.append(loss.numpy().tolist())
            auc, acc, rms = get_auc(tf.reshape(y_test, [-1]).numpy().tolist(),
                                    tf.reshape(y_pred, [-1]).numpy().tolist(),
                                    pos_ratio)
            auc_history.append(auc)
            acc_history.append(acc)
            rms_history.append(rms)
            info += f'{model.name[:-2]+" "*(12-len(model.name))}\t{loss:4f}\t{rms:4f}\t{auc:4f}\t{acc:4f}\n'
        print(info[:-1])
    step += 1

print('Final result')
info = 'model     \tloss    \trms     \tauc     \tacc     \n'
for model, l1, l2, l3, l4 in zip(model_list,
                                 train_loss_history_list,
                                 train_rms_history_list,
                                 train_auc_history_list,
                                 train_acc_history_list):
    n = model.name
    n = '_'.join(n.split('_')[:-1])
    info += f'{n + " " * (10 - len(n))}\t{min(l1):4f}\t{min(l2):4f}\t{max(l3):4f}\t{max(l4):4f}\n'
print(info)

