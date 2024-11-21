import tensorflow as tf
import hashlib

dense_dim=33
r_dim=4
hidden_dim=36

def rms_norm(x):
    return x / (tf.norm(x, axis=-1, keepdims=True) + 1e-5)

def get_variable(shape, name):
    if len(shape) == 2:
        l = 12. / (shape[0] + shape[1])
        res = tf.Variable(tf.random.uniform(shape, -l, l), name=name)
        return res
    else:
        return tf.Variable(tf.zeros(shape), name=name)

def get_embedding(shape, name):
    res = tf.Variable(tf.random.normal(shape, stddev=0.001), name=name)
    return res

class MlpModel(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed = int(hashlib.md5(self.name.encode()).hexdigest(), 16) & (2**32-1)
        with tf.name_scope(f'{self.name}_net'):
            tf.random.set_seed(self.seed)
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.wr = get_variable([r_dim, hidden_dim], 'wr')
            self.b = get_variable([hidden_dim], 'b')
            self.w1 = get_variable([hidden_dim, hidden_dim], 'w1')
            self.b1 = get_variable([hidden_dim], 'b1')
            self.w2 = get_variable([hidden_dim, 1], 'w2')
            self.b2 = get_variable([1], 'b2')
    def __call__(self, inputs, y, **kwargs):
        x, r = inputs
        with tf.name_scope(f'{self.name}_net'):
            s = x @ self.wx + r @ self.wr + self.b
            s = tf.tanh(s)
            s = s @ self.w1 + self.b1
            s = tf.tanh(s)
            s = s @ self.w2 + self.b2
            y_pred = tf.squeeze(tf.nn.sigmoid(s), -1)
            loss = tf.reduce_mean(- y * tf.math.log(y_pred + 1e-5) - (1 - y) * tf.math.log(1 - y_pred + 1e-5))
            return y_pred, loss


class MinmaxModel(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed = int(hashlib.md5(self.name.encode()).hexdigest(), 16) & (2**32-1)
        with tf.name_scope(f'{self.name}_net'):
            tf.random.set_seed(self.seed)
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.wr = get_variable([r_dim, hidden_dim], 'wr')
            self.b = get_variable([hidden_dim], 'b')
            self.w1 = get_variable([hidden_dim, hidden_dim], 'w1')
            self.b1 = get_variable([hidden_dim], 'b1')
            self.w2 = get_variable([hidden_dim, 9], 'w2')
            self.b2 = get_variable([9], 'b2')
    def __call__(self, inputs, y, **kwargs):
        x, r = inputs
        with tf.name_scope(f'{self.name}_net'):
            s = x @ self.wx + r @ tf.nn.softplus(10 * self.wr) / 10 + self.b
            s = tf.nn.tanh(s)
            s = s @ tf.nn.softplus(self.w1 * 10) / 10 + self.b1
            s = tf.nn.tanh(s)
            s = s @ tf.nn.softplus(self.w2 * 10) / 10 + self.b2
            s = tf.reshape(s, [-1, 3, 3])
            s = tf.reduce_min(s, -1)
            s = tf.reduce_max(s, -1)
            y_pred = tf.nn.sigmoid(s)
            loss = tf.reduce_mean(- y * tf.math.log(y_pred + 1e-5) - (1 - y) * tf.math.log(1 - y_pred + 1e-5))
            return y_pred, loss

class SmoothedMinmaxModel(tf.Module):
    def __init__(self, beta=4, **kwargs):
        super().__init__(**kwargs)
        self.seed = int(hashlib.md5(self.name.encode()).hexdigest(), 16) & (2**32-1)
        with tf.name_scope(f'{self.name}_net'):
            tf.random.set_seed(self.seed)
            self.beta = beta
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.wr = get_variable([r_dim, hidden_dim], 'wr')
            self.b = get_variable([hidden_dim], 'b')
            self.w1 = get_variable([hidden_dim, hidden_dim], 'w1')
            self.b1 = get_variable([hidden_dim], 'b1')
            self.w2 = get_variable([hidden_dim, 9], 'w2')
            self.b2 = get_variable([9], 'b2')
    def __call__(self, inputs, y, **kwargs):
        x, r = inputs
        with tf.name_scope(f'{self.name}_net'):
            s = x @ self.wx + r @ tf.nn.softplus(10 * self.wr) / 10 + self.b
            s = tf.nn.tanh(s)
            s = s @ tf.nn.softplus(self.w1 * 10) / 10 + self.b1
            s = tf.nn.tanh(s)
            s = s @ tf.nn.softplus(self.w2 * 10) / 10 + self.b2
            s = tf.reshape(s, [-1, 3, 3])
            s = - tf.reduce_logsumexp(-s * self.beta, -1) / self.beta
            s = tf.reduce_logsumexp(s * self.beta, -1) / self.beta
            y_pred = tf.nn.sigmoid(s)
            loss = tf.reduce_mean(- y * tf.math.log(y_pred + 1e-5) - (1 - y) * tf.math.log(1 - y_pred + 1e-5))
            return y_pred, loss

class ConstrainedModel(tf.Module):
    def __init__(self, beta=4, **kwargs):
        super().__init__(**kwargs)
        self.seed = int(hashlib.md5(self.name.encode()).hexdigest(), 16) & (2**32-1)
        with tf.name_scope(f'{self.name}_net'):
            tf.random.set_seed(self.seed)
            self.beta = beta
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.wr = get_variable([r_dim, hidden_dim], 'wr')
            self.b = get_variable([hidden_dim], 'b')
            self.w1 = get_variable([hidden_dim, hidden_dim], 'w1')
            self.b1 = get_variable([hidden_dim], 'b1')
            self.w2 = get_variable([hidden_dim, 1], 'w2')
            self.b2 = get_variable([1], 'b2')
    def __call__(self, inputs, y, **kwargs):
        x, r = inputs
        with tf.name_scope(f'{self.name}_net'):
            f1 = lambda x: tf.nn.softplus(x)
            f2 = lambda x: -tf.nn.softplus(-x)
            f3 = lambda x: f2(tf.nn.relu(x)-1) + f1(-tf.nn.relu(-x)+1)
            s = x @ self.wx + r @ tf.nn.softplus(10 * self.wr) / 10 + self.b
            t = int(hidden_dim/3)
            s = tf.concat([f1(s[:, :t]), f2(s[:, t:2*t]), f3(s[:, 2*t:])], -1)
            s = s @ tf.nn.softplus(self.w1 * 10) / 10 + self.b1
            s = tf.concat([f1(s[:, :t]), f2(s[:, t:2*t]), f3(s[:, 2*t:])], -1)
            s = s @ tf.nn.softplus(self.w2 * 10) / 10 + self.b2
            s = tf.squeeze(s, -1)
            y_pred = tf.nn.sigmoid(s)
            loss = tf.reduce_mean(- y * tf.math.log(y_pred + 1e-5) - (1 - y) * tf.math.log(1 - y_pred + 1e-5))
            return y_pred, loss

class LatticeModel(tf.Module):
    def __init__(self, k_list=[2]*r_dim, **kwargs):
        super().__init__(**kwargs)
        self.seed = int(hashlib.md5(self.name.encode()).hexdigest(), 16) & (2 ** 32 - 1)
        with tf.name_scope(f'{self.name}_net'):
            tf.random.set_seed(self.seed)
            self.k_list = k_list
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.b = get_variable([hidden_dim], 'b')
            self.w1 = get_variable([hidden_dim, hidden_dim], 'w1')
            self.b1 = get_variable([hidden_dim], 'b1')
            self.w2 = get_variable([hidden_dim, 1], 'w2')
            self.b2 = get_variable([1], 'b2')
            lattice = tf.Variable(tf.zeros(k_list), name='lattice')
            self.lattice = lattice
            indices = [[]]
            for k in k_list:
                tmp = []
                for i in range(k):
                    for j in indices:
                        tmp.append(j + [i])
                indices = tmp
            self.indices = tf.constant(indices, dtype=tf.float32)

    def __call__(self, inputs, y, **kwargs):
        x, r = inputs
        with tf.name_scope(f'{self.name}_net'):
            idx, lattice, k_list = self.indices, self.lattice, self.k_list
            lattice = tf.math.softplus(lattice)
            for k in k_list:
                add_mat = tf.constant([[0.] * i + [1.] * (k-i)  for i in range(k)])
                lattice = tf.transpose(lattice, [i % r_dim for i in range(1, r_dim+1)])
                lattice = lattice @ add_mat
            r_weight = tf.expand_dims(r*(tf.constant(k_list, tf.float32)-1), 1) - tf.expand_dims(idx, 0)
            r_weight = 1 - tf.abs(r_weight)
            r_weight = tf.where(tf.greater(r_weight, 0), r_weight, tf.zeros_like(r_weight))
            r_weight = tf.where(tf.less_equal(r_weight, 1), r_weight, tf.zeros_like(r_weight))
            tmp = 1
            for i in range(r_dim):
                tmp *= r_weight[:,:,i]
            r_weight = tmp
            r_weight = tf.reshape(r_weight, [-1] + k_list)
            r_weight *= tf.expand_dims(lattice, 0)
            tmp = 1
            for i in range(r_dim):
                tmp *= k_list[i]
            r_weight = tf.reshape(r_weight, [-1, tmp])
            r_weight = tf.reduce_sum(r_weight, -1, keepdims=True)
            self.r_weight = r_weight
            s = x @ self.wx + self.b
            s = tf.tanh(s)
            s = s @ self.w1 + self.b1
            s = tf.tanh(s)
            s = s @ self.w2 + self.b2
            s += r_weight
            y_pred = tf.squeeze(tf.nn.sigmoid(s), -1)
            loss = tf.reduce_mean(- y * tf.math.log(y_pred + 1e-5) - (1 - y) * tf.math.log(1 - y_pred + 1e-5))
            return y_pred, loss


class HintModel(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed = int(hashlib.md5(self.name.encode()).hexdigest(), 16) & (2**32-1)
        with tf.name_scope(f'{self.name}_net'):
            tf.random.set_seed(self.seed)
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.wr = get_variable([r_dim, hidden_dim], 'wr')
            self.b = get_variable([hidden_dim], 'b')
            self.w1 = get_variable([hidden_dim, hidden_dim], 'w1')
            self.b1 = get_variable([hidden_dim], 'b1')
            self.w2 = get_variable([hidden_dim, 1], 'w2')
            self.b2 = get_variable([1], 'b2')
    def __call__(self, inputs, y, **kwargs):
        x, r = inputs
        with tf.name_scope(f'{self.name}_net'):
            s = x @ self.wx + r @ self.wr + self.b
            s = tf.tanh(s)
            s = s @ self.w1 + self.b1
            s = tf.tanh(s)
            s = s @ self.w2 + self.b2
            y_pred = tf.squeeze(tf.nn.sigmoid(s), -1)
            r_ = r + tf.random.uniform(r.shape) * 0.1
            s_ = x @ self.wx + r_ @ self.wr + self.b
            s_ = tf.tanh(s_)
            s_ = s_ @ self.w1 + self.b1
            s_ = tf.tanh(s_)
            s_ = s_ @ self.w2 + self.b2
            delta = tf.reduce_mean(tf.nn.relu(s-s_) ** 2)
            delta -= tf.stop_gradient(delta)
            loss = tf.reduce_mean(- y * tf.math.log(y_pred + 1e-5) - (1 - y) * tf.math.log(1 - y_pred + 1e-5)) + delta
            return y_pred, loss

class PwlModel(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed = int(hashlib.md5(self.name.encode()).hexdigest(), 16) & (2**32-1)
        with tf.name_scope(f'{self.name}_net'):
            tf.random.set_seed(self.seed)
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.wr = get_variable([r_dim, hidden_dim], 'wr')
            self.b = get_variable([hidden_dim], 'b')
            self.w1 = get_variable([hidden_dim, hidden_dim], 'w1')
            self.b1 = get_variable([hidden_dim], 'b1')
            self.w2 = get_variable([hidden_dim, 1], 'w2')
            self.b2 = get_variable([1], 'b2')
    def __call__(self, inputs, y, **kwargs):
        x, r = inputs
        with tf.name_scope(f'{self.name}_net'):
            h1 = x @ self.wx + r @ self.wr + self.b
            h1 = tf.tanh(h1)
            h2 = h1 @ self.w1 + self.b1
            h2 = tf.tanh(h2)
            s = h2 @ self.w2 + self.b2
            y_pred = tf.squeeze(tf.nn.sigmoid(s), -1)
            loss = tf.reduce_mean(- y * tf.math.log(y_pred + 1e-5) - (1 - y) * tf.math.log(1 - y_pred + 1e-5))
            h2_grads = (tf.transpose(self.w2) * (1-tf.square(tf.stop_gradient(h2)))) @ tf.transpose(self.w1)
            r_grads = (h2_grads * (1-tf.square(tf.stop_gradient(h1)))) @ tf.transpose(self.wr)
            reg = tf.reduce_sum(tf.nn.relu(-r_grads)) * 0.01
            reg -= tf.stop_gradient(reg)
            loss += reg
        return y_pred, loss

class GenCostModel(tf.Module):
    def __init__(self, sample_num = 64, kl_ratio = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.seed = int(hashlib.md5(self.name.encode()).hexdigest(), 16) & (2**32-1)
        with tf.name_scope(f'{self.name}_net'):
            tf.random.set_seed(self.seed)
            self.kl_ratio = kl_ratio
            self.sample_num = sample_num
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.b = get_variable([hidden_dim], 'b')
            self.wp = get_variable([hidden_dim, sample_num], 'wp')
            self.wh = get_variable([hidden_dim, hidden_dim], 'wh')
            self.bh = get_variable([hidden_dim], 'bh')
            self.wc = get_variable([hidden_dim, r_dim], 'wc')
            self.bc = get_variable([r_dim], 'bc')
            self.log_sig_out = get_variable([r_dim], 'sig_out')
            self.base = get_variable([sample_num, hidden_dim], 'base')
    def __call__(self, inputs, y, sample_num=1, same_sample=False, **kwargs):
        x, r = inputs
        with tf.name_scope(f'{self.name}_net'):
            x = x @ self.wx + self.b
            x = tf.nn.tanh(x)
            proj_prob = tf.nn.softmax(x @ self.wp, -1)
            kl = tf.reduce_mean(tf.reduce_sum(proj_prob * tf.math.log(proj_prob * self.sample_num), -1))
            z = tf.expand_dims(x, 1) + tf.expand_dims(self.base, 0)
            h = z @ self.wh + self.bh
            h = tf.nn.tanh(h)
            mu_out = h @ self.wc + self.bc
            sig_out = tf.math.exp(self.log_sig_out)
            s = (tf.expand_dims(r, 1) - mu_out) / sig_out
            prob = tf.nn.sigmoid(s)
            y_pred = tf.exp(tf.reduce_sum(tf.math.log(prob), -1))
            y_pred = tf.reduce_sum(y_pred * proj_prob, 1)
            loss = tf.reduce_mean(- y * tf.math.log(y_pred + 1e-5) - (1 - y) * tf.math.log(1 - y_pred + 1e-5))
            loss += (kl - tf.stop_gradient(kl)) * self.kl_ratio
            return y_pred, loss

class GcmViModel(tf.Module):
    def __init__(self, sample_num = 64, kl_ratio = 0.1, elb_ratio = 0.1, ll_ratio=1, **kwargs):
        super().__init__(**kwargs)
        self.seed = int(hashlib.md5(self.name.encode()).hexdigest(), 16) & (2**32-1)
        with tf.name_scope(f'{self.name}_net'):
            tf.random.set_seed(self.seed)
            self.kl_ratio = kl_ratio
            self.ll_ratio = ll_ratio
            self.sample_num = sample_num
            self.elb_ratio = elb_ratio
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.b = get_variable([hidden_dim], 'b')
            self.wp = get_variable([hidden_dim, sample_num], 'wp')
            self._wx = get_variable([dense_dim+r_dim, hidden_dim], '_wx')
            self._b = get_variable([hidden_dim], '_b')
            self._wx2 = get_variable([hidden_dim, hidden_dim], '_wx2')
            self._b2 = get_variable([hidden_dim], '_b2')
            self._wp = get_variable([hidden_dim, sample_num], '_wp')
            self.wh = get_variable([hidden_dim, hidden_dim], 'wh')
            self.bh = get_variable([hidden_dim], 'bh')
            self.wc = get_variable([hidden_dim, r_dim], 'wc')
            self.bc = get_variable([r_dim], 'bc')
            self.log_sig_out = get_variable([r_dim], 'sig_out')
            self.base = get_variable([sample_num, hidden_dim], 'base')
    def __call__(self, inputs, y, sample_num=1, same_sample=False, **kwargs):
        x, r = inputs
        with tf.name_scope(f'{self.name}_net'):
            h = x @ self.wx + self.b
            h = tf.nn.tanh(h)
            proj_prob = tf.nn.softmax(h @ self.wp, -1)
            kl = tf.reduce_mean(tf.reduce_sum(proj_prob * tf.math.log(proj_prob * self.sample_num), -1))
            z = tf.expand_dims(h, 1) + tf.expand_dims(self.base, 0)
            h = z @ self.wh + self.bh
            h = tf.nn.tanh(h)
            mu_out = h @ self.wc + self.bc
            sig_out = tf.math.exp(self.log_sig_out)
            s = (tf.expand_dims(r, 1) - mu_out) / sig_out
            prob = tf.nn.sigmoid(s)
            prob = tf.exp(tf.reduce_sum(tf.math.log(prob), -1))
            y_pred = tf.reduce_sum(prob * proj_prob, 1)
            loss = tf.reduce_mean(- y * tf.math.log(y_pred + 1e-5) - (1 - y) * tf.math.log(1 - y_pred + 1e-5))
            loss *= self.ll_ratio
            #
            _y = tf.expand_dims(y, -1)
            _h = tf.concat([x, r], -1) @ self._wx + self._b
            _h = tf.nn.tanh(_h)
            _h = _h @ self._wx2 + self._b2
            _h = tf.nn.tanh(_h)
            _proj_prob = tf.nn.softmax(_h @ self._wp, -1)
            _loss = tf.reduce_mean(- _y * tf.math.log(prob + 1e-5) - (1 - _y) * tf.math.log(1 - prob + 1e-5))
            _loss = tf.reduce_mean(tf.reduce_sum(_loss * _proj_prob, 1))
            if self.ll_ratio > 0:
                _kl = tf.reduce_mean(tf.reduce_sum(_proj_prob * (tf.math.log(_proj_prob)
                                                                 - tf.math.log(tf.stop_gradient(proj_prob))), -1))
            else:
                _kl = tf.reduce_mean(tf.reduce_sum(_proj_prob * (tf.math.log(_proj_prob)
                                                                 - tf.math.log(proj_prob)), -1))
            _elb = _kl + _loss
            _elb -= tf.stop_gradient(_elb)
            loss += _elb * self.elb_ratio
            loss += (kl - tf.stop_gradient(kl)) * self.kl_ratio
            return y_pred, loss

class GcmViRevModel(tf.Module):
    def __init__(self, sample_num = 64, kl_ratio = 0.1, elb_ratio = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.seed = int(hashlib.md5(self.name.encode()).hexdigest(), 16) & (2**32-1)
        with tf.name_scope(f'{self.name}_net'):
            tf.random.set_seed(self.seed)
            self.kl_ratio = kl_ratio
            self.sample_num = sample_num
            self.elb_ratio = elb_ratio
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.b = get_variable([hidden_dim], 'b')
            self.wp = get_variable([hidden_dim, sample_num], 'wp')
            self._wx = get_variable([dense_dim+r_dim, hidden_dim], '_wx')
            self._b = get_variable([hidden_dim], '_b')
            self._wx2 = get_variable([hidden_dim, hidden_dim], '_wx2')
            self._b2 = get_variable([hidden_dim], '_b2')
            self._wp = get_variable([hidden_dim, sample_num], '_wp')
            self.wh = get_variable([hidden_dim, hidden_dim], 'wh')
            self.bh = get_variable([hidden_dim], 'bh')
            self.wc = get_variable([hidden_dim, r_dim], 'wc')
            self.bc = get_variable([r_dim], 'bc')
            self.log_sig_c = get_variable([r_dim], 'sig_c')
            self.wr = get_variable([hidden_dim, r_dim], 'wr')
            self.br = get_variable([r_dim], 'br')
            self.base = get_variable([sample_num, hidden_dim], 'base')
    def __call__(self, inputs, y, sample_num=1, same_sample=False, **kwargs):
        x, r = inputs
        with tf.name_scope(f'{self.name}_net'):
            h = x @ self.wx + self.b
            h = tf.nn.tanh(h)
            proj_prob = tf.nn.softmax(h @ self.wp, -1) 
            kl = tf.reduce_mean(tf.reduce_sum(proj_prob * tf.math.log(proj_prob * self.sample_num), -1))
            z = tf.expand_dims(h, 1) + tf.expand_dims(self.base, 0) 
            h = z @ self.wh + self.bh
            h = tf.nn.tanh(h)
            mu_r = tf.nn.sigmoid(h @ self.wr + self.br)
            r_loss = tf.reduce_sum(0.5 * tf.square((tf.expand_dims(r, 1) - mu_r)), -1)
            r_loss = tf.reduce_mean(tf.reduce_sum(r_loss * tf.stop_gradient(proj_prob), -1))
            r_loss -= tf.stop_gradient(r_loss)
            mu_c = h @ self.wc + self.bc
            sig_c = tf.math.exp(self.log_sig_c)
            s = (tf.expand_dims(r, 1) - mu_c) / sig_c
            prob = tf.nn.sigmoid(s)
            prob = tf.exp(tf.reduce_sum(tf.math.log(prob), -1))  # [b, s]
            y_pred = tf.reduce_sum(prob * proj_prob, 1)
            loss = tf.reduce_mean(- y * tf.math.log(y_pred + 1e-5) - (1 - y) * tf.math.log(1 - y_pred + 1e-5))
            #
            _y = tf.expand_dims(y, -1)  # [b, 1]
            _h = tf.concat([x, r], -1) @ self._wx + self._b
            _h = tf.nn.tanh(_h)
            _h = _h @ self._wx2 + self._b2
            _h = tf.nn.tanh(_h)
            _proj_prob = tf.nn.softmax(_h @ self._wp, -1)  # [b, h]
            _loss = tf.reduce_mean(- _y * tf.math.log(prob + 1e-5) - (1 - _y) * tf.math.log(1 - prob + 1e-5))  # [b, h]
            _loss = tf.reduce_mean(tf.reduce_sum(_loss * _proj_prob, 1))
            _kl = tf.reduce_mean(tf.reduce_sum(_proj_prob * (tf.math.log(_proj_prob) -
                                                             tf.math.log(tf.stop_gradient(proj_prob))), -1))
            _r_loss = tf.reduce_mean(tf.reduce_sum(r_loss * tf.stop_gradient(_proj_prob), -1))
            _r_loss -= tf.stop_gradient(r_loss)
            _elb = _kl + _loss + _r_loss
            _elb -= tf.stop_gradient(_elb)
            loss += _elb * self.elb_ratio
            loss += (kl - tf.stop_gradient(kl)) * self.kl_ratio
            return y_pred, loss
