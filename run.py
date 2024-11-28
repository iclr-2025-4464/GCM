import sys
from model import *
from data_loader import DataLoader

datasets = ['Adult', 'Diabetes', 'BlogFeedback']
def train_step(x, r, y, model, lr = 0.001, reg_ratio = 1e-6, is_binary=True):
    vars = model.variables
    with tf.GradientTape() as t:
        l2_reg = sum([tf.reduce_sum(tf.square(var)) for var in vars])
        y_pred, loss = model([x, r], y=y)
        total_loss = loss + l2_reg * reg_ratio
    if lr > 0:
        grads = t.gradient(total_loss, vars)
        grads = [tf.convert_to_tensor(i) for i in grads]
        for grad, var in zip(grads, vars):
            if is_binary:
                var.assign_sub(lr * tf.clip_by_value(grad, -100, 100))
            else:
                var.assign_sub(lr * tf.clip_by_value(grad, -1, 1))
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

data_loader = DataLoader()
def run_exp(dataset_id, seed):
    x_train, r_train, y_train, x_test, r_test, y_test, dense_dim, r_dim, is_binary = data_loader.load(datasets[dataset_id], seed)
    model_list = [
        MlpModel(name=f'MLP_{seed}', dense_dim=dense_dim, r_dim=r_dim, is_binary=is_binary),
        MinmaxModel(name=f'MM_{seed}', dense_dim=dense_dim, r_dim=r_dim, is_binary=is_binary),
        SmoothedMinmaxModel(name=f'SMM_{seed}', dense_dim=dense_dim, r_dim=r_dim, is_binary=is_binary),
        ConstrainedModel(name=f'CMNN_{seed}', dense_dim=dense_dim, r_dim=r_dim, is_binary=is_binary),
        HintModel(name=f'Hint_{seed}', dense_dim=dense_dim, r_dim=r_dim, is_binary=is_binary),
        PwlModel(name=f'PWL_{seed}', dense_dim=dense_dim, r_dim=r_dim, is_binary=is_binary),
        GcmModel(name=f'GCM_{seed}', dense_dim=dense_dim, r_dim=r_dim, is_binary=is_binary),
        GcmViModel(name=f'GCM_VI_{seed}', dense_dim=dense_dim, r_dim=r_dim, is_binary=is_binary),
        GcrmViModel(name=f'GCRM_VI_{seed}', dense_dim=dense_dim, r_dim=r_dim, is_binary=is_binary),
        GcmCateModel(name=f'GCM_cate_{seed}', dense_dim=dense_dim, r_dim=r_dim, is_binary=is_binary),
    ]
    train_loss_history_list = [[] for _ in model_list]
    train_auc_history_list = [[] for _ in model_list]
    train_acc_history_list = [[] for _ in model_list]
    train_rms_history_list = [[] for _ in model_list]
    report_interval = 100
    step = 0
    N = 100
    total_step = N * report_interval
    lr = 0.3 if is_binary else 0.03
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
                _, _ = train_step(x, r, y, model, lr, is_binary=is_binary)
            lr *= 0.9997
        else:
            info = f'step: {step} lr: {lr:.4f}\nmodel     \tloss    \trms     \tauc     \tacc     \n'
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
                n = model.name
                n = '_'.join(n.split('_')[:-1])
                info += f'{n+" "*(10-len(n))}\t{loss:.6f}\t{rms:.6f}\t{auc:.6f}\t{acc:.6f}\n'
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
        info += f'{n + " " * (10 - len(n))}\t{min(l1):.6f}\t{min(l2):.6f}\t{max(l3):.6f}\t{max(l4):.6f}\n'
    print(info)


if __name__ == '__main__':
    args = sys.argv
    if len(args) > 1:
        dataset_id = int(sys.argv[1])
    else:
        dataset_id = 0
    if len(args) > 2:
        seed = int(sys.argv[2])
    else:
        seed = 0
    print(f'dataset: {datasets[dataset_id]}')
    print(f'seed: {seed}')
    run_exp(dataset_id, seed)

