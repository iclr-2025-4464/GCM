import tensorflow as tf
import hashlib

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

def pos_w(w, k=10.0):
    return tf.nn.softplus(w * k) / k

def ce_loss(y, y_pred):
    return tf.reduce_mean(- y * tf.math.log(y_pred + 1e-5) - (1-y) * tf.math.log(1 - y_pred + 1e-5))

def mse_loss(y, y_pred):
    return tf.reduce_mean(tf.square(y_pred-y))

class BaseModel(tf.Module):
    def __init__(self, dense_dim=33, r_dim=4, hidden_dim=16, is_binary=1, **kwargs):
        super().__init__(**kwargs)
        self.seed = int(hashlib.md5(self.name.encode()).hexdigest(), 16) & (2**32-1)
        tf.random.set_seed(self.seed)
        self.dense_dim = dense_dim
        self.r_dim = r_dim
        self.hidden_dim = hidden_dim
        self.is_binary = is_binary


class MlpModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dense_dim, r_dim, hidden_dim = self.dense_dim, self.r_dim, self.hidden_dim
        with tf.name_scope(f'{self.name}_net'):
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
            s = tf.squeeze(s, -1)
            if self.is_binary:
                y_pred = tf.nn.sigmoid(s)
                loss = ce_loss(y, y_pred)
            else:
                y_pred = s
                loss = mse_loss(y, y_pred)
            return y_pred, loss


class MinmaxModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dense_dim, r_dim, hidden_dim = self.dense_dim, self.r_dim, self.hidden_dim
        with tf.name_scope(f'{self.name}_net'):
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.bx = get_variable([hidden_dim], 'bx')
            self.wx1 = get_variable([hidden_dim, hidden_dim], 'wx1')
            self.bx1 = get_variable([hidden_dim], 'bx1')
            self.wx2 = get_variable([hidden_dim, 9], 'wx2')
            self.b2 = get_variable([9], 'b2')
            self.wr = get_variable([r_dim, hidden_dim], 'wr')
            self.br = get_variable([hidden_dim], 'br')
            self.wr1 = get_variable([hidden_dim, hidden_dim], 'wr1')
            self.br1 = get_variable([hidden_dim], 'br1')
            self.wr2 = get_variable([hidden_dim, 9], 'wr2')
    def __call__(self, inputs, y, **kwargs):
        x, r = inputs
        with tf.name_scope(f'{self.name}_net'):
            x = x @ self.wx + self.bx
            xr = r @ pos_w(self.wr) + self.br + x
            x = tf.nn.tanh(x)
            xr = tf.nn.tanh(xr)
            x = x @ self.wx1 + self.bx1
            xr = xr @ pos_w(self.wr1) + self.br1 + x
            x = tf.nn.tanh(x)
            xr = tf.nn.tanh(xr)
            s = xr @ pos_w(self.wr2) + x @ self.wx2 + self.b2
            s = tf.reshape(s, [-1, 3, 3])
            s = tf.reduce_min(s, -1)
            s = tf.reduce_max(s, -1)
            if self.is_binary:
                y_pred = tf.nn.sigmoid(s)
                loss = ce_loss(y, y_pred)
            else:
                y_pred = s
                loss = mse_loss(y, y_pred)
            return y_pred, loss

class SmoothedMinmaxModel(BaseModel):
    def __init__(self, beta=4, **kwargs):
        super().__init__(**kwargs)
        dense_dim, r_dim, hidden_dim = self.dense_dim, self.r_dim, self.hidden_dim
        with tf.name_scope(f'{self.name}_net'):
            self.beta = beta
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.bx = get_variable([hidden_dim], 'bx')
            self.wx1 = get_variable([hidden_dim, hidden_dim], 'wx1')
            self.bx1 = get_variable([hidden_dim], 'bx1')
            self.wx2 = get_variable([hidden_dim, 9], 'wx2')
            self.b2 = get_variable([9], 'b2')
            self.wr = get_variable([r_dim, hidden_dim], 'wr')
            self.br = get_variable([hidden_dim], 'br')
            self.wr1 = get_variable([hidden_dim, hidden_dim], 'wr1')
            self.br1 = get_variable([hidden_dim], 'br1')
            self.wr2 = get_variable([hidden_dim, 9], 'wr2')
    def __call__(self, inputs, y, **kwargs):
        x, r = inputs
        with tf.name_scope(f'{self.name}_net'):
            x = x @ self.wx + self.bx
            xr = r @ pos_w(self.wr) + self.br + x
            x = tf.nn.tanh(x)
            xr = tf.nn.tanh(xr)
            x = x @ self.wx1 + self.bx1
            xr = xr @ pos_w(self.wr1) + self.br1 + x
            x = tf.nn.tanh(x)
            xr = tf.nn.tanh(xr)
            s = xr @ pos_w(self.wr2) + x @ self.wx2 + self.b2
            s = tf.reshape(s, [-1, 3, 3])
            s = - tf.reduce_logsumexp(-s * self.beta, -1) / self.beta
            s = tf.reduce_logsumexp(s * self.beta, -1) / self.beta
            if self.is_binary:
                y_pred = tf.nn.sigmoid(s)
                loss = ce_loss(y, y_pred)
            else:
                y_pred = s
                loss = mse_loss(y, y_pred)
            return y_pred, loss

class ConstrainedModel(BaseModel):
    def __init__(self, beta=4, **kwargs):
        super().__init__(**kwargs)
        dense_dim, r_dim, hidden_dim = self.dense_dim, self.r_dim, self.hidden_dim
        with tf.name_scope(f'{self.name}_net'):
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.bx = get_variable([hidden_dim], 'bx')
            self.wx1 = get_variable([hidden_dim, hidden_dim], 'wx1')
            self.bx1 = get_variable([hidden_dim], 'bx1')
            self.wx2 = get_variable([hidden_dim, 1], 'wx2')
            self.b2 = get_variable([1], 'b2')
            self.wr = get_variable([r_dim, hidden_dim], 'wr')
            self.br = get_variable([hidden_dim], 'br')
            self.wr1 = get_variable([hidden_dim, hidden_dim], 'wr1')
            self.br1 = get_variable([hidden_dim], 'br1')
            self.wr2 = get_variable([hidden_dim, 1], 'wr2')
    def __call__(self, inputs, y, **kwargs):
        x, r = inputs
        with tf.name_scope(f'{self.name}_net'):
            f1 = lambda x: tf.nn.softplus(x)
            f2 = lambda x: -tf.nn.softplus(-x)
            f3 = lambda x: f2(tf.nn.relu(x)-1) + f1(-tf.nn.relu(-x)+1)
            t = int(self.hidden_dim/3)
            fr = lambda x: tf.concat([f1(x[:, :t]), f2(x[:, t:2*t]), f3(x[:, 2*t:])], -1)
            x = x @ self.wx + self.bx
            xr = r @ pos_w(self.wr) + self.br + x
            x = tf.nn.tanh(x)
            xr = fr(xr)
            x = x @ self.wx1 + self.bx1
            xr = xr @ pos_w(self.wr1) + self.br1 + x
            x = tf.nn.tanh(x)
            xr = fr(xr)
            s = x @ self.wx2 + xr @ pos_w(self.wr2) + self.b2
            s = tf.squeeze(s, -1)
            if self.is_binary:
                y_pred = tf.nn.sigmoid(s)
                loss = ce_loss(y, y_pred)
            else:
                y_pred = s
                loss = mse_loss(y, y_pred)
            return y_pred, loss

class HintModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dense_dim, r_dim, hidden_dim = self.dense_dim, self.r_dim, self.hidden_dim
        with tf.name_scope(f'{self.name}_net'):
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
            s = tf.squeeze(s, -1)
            if self.is_binary:
                y_pred = tf.nn.sigmoid(s)
                loss = ce_loss(y, y_pred)
            else:
                y_pred = s
                loss = mse_loss(y, y_pred)
            r_ = r + tf.random.uniform(r.shape) * 0.1
            s_ = x @ self.wx + r_ @ self.wr + self.b
            s_ = tf.tanh(s_)
            s_ = s_ @ self.w1 + self.b1
            s_ = tf.tanh(s_)
            s_ = s_ @ self.w2 + self.b2
            delta = tf.reduce_mean(tf.nn.relu(s-s_) ** 2)
            delta -= tf.stop_gradient(delta)
            loss += delta
            return y_pred, loss

class PwlModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dense_dim, r_dim, hidden_dim = self.dense_dim, self.r_dim, self.hidden_dim
        with tf.name_scope(f'{self.name}_net'):
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
            s = tf.squeeze(s, -1)
            if self.is_binary:
                y_pred = tf.nn.sigmoid(s)
                loss = ce_loss(y, y_pred)
            else:
                y_pred = s
                loss = mse_loss(y, y_pred)
            h2_grads = (tf.transpose(self.w2) * (1-tf.square(tf.stop_gradient(h2)))) @ tf.transpose(self.w1)
            r_grads = (h2_grads * (1-tf.square(tf.stop_gradient(h1)))) @ tf.transpose(self.wr)
            reg = tf.reduce_sum(tf.nn.relu(-r_grads)) * 0.01
            reg -= tf.stop_gradient(reg)
            loss += reg
        return y_pred, loss

class GcmModel(BaseModel):
    def __init__(self, sample_num = 32, z_dim=4, **kwargs):
        super().__init__(**kwargs)
        dense_dim, r_dim, hidden_dim = self.dense_dim, self.r_dim, self.hidden_dim
        with tf.name_scope(f'{self.name}_net'):
            self.sample_num = sample_num
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.b = get_variable([hidden_dim], 'b')
            self.wmu = get_variable([hidden_dim, z_dim], 'wmu')
            self.bmu = get_variable([z_dim], 'bmu')
            self.wsig = get_variable([hidden_dim, z_dim], 'wsig')
            self.bsig = get_variable([z_dim], 'bsig')
            self.wh = get_variable([z_dim, hidden_dim], 'wh')
            self.bh = get_variable([hidden_dim], 'bh')
            self.wmc = get_variable([hidden_dim, r_dim], 'wmc')
            self.bmc = get_variable([r_dim], 'bmc')
            self.wsc = get_variable([hidden_dim, r_dim], 'wsc')
            self.bsc = get_variable([r_dim], 'bsc')
            if not self.is_binary:
                self.wt1 = get_variable([z_dim, hidden_dim], 'wt1')
                self.bt1 = get_variable([hidden_dim], 'bt1')
                self.wt2 = get_variable([hidden_dim, 1], 'wt2')
                self.bt2 = get_variable([1], 'bt2')
                self.wsy1 = get_variable([z_dim, hidden_dim], 'wsy1')
                self.bsy1 = get_variable([hidden_dim], 'bsy1')
                self.wsy2 = get_variable([hidden_dim, 1], 'wsy2')
                self.bsy2 = get_variable([1], 'bsy2')
    def __call__(self, inputs, y, is_test=False, **kwargs):
        x, r = inputs
        sample_num = self.sample_num
        with tf.name_scope(f'{self.name}_net'):
            h = x @ self.wx + self.b
            h = tf.nn.tanh(h)
            mu = h @ self.wmu + self.bmu  # [b, d]
            log_var = h @ self.wsig + self.bsig  # [b, d]
            mu = tf.tile(tf.expand_dims(mu, 1), [1, sample_num, 1])
            sig = tf.tile(tf.expand_dims(tf.math.exp(0.5 * log_var), 1), [1, sample_num, 1])
            z = tf.random.truncated_normal(tf.shape(mu)) * sig + mu  # [b, k, d]  z ~ p(z|x)
            h = z @ self.wh + self.bh
            h = tf.nn.tanh(h)  # [b, k, d]
            mu_c = h @ self.wmc + self.bmc
            s_c = tf.nn.softplus(h @ self.wsc + self.bsc) + 1e-2
            s = (tf.expand_dims(r, 1) - mu_c) / s_c  # [b, k, nr]
            prob = tf.nn.sigmoid(s)
            prob = tf.exp(tf.reduce_sum(tf.math.log(prob), -1))  # [b, k]
            if self.is_binary:
                y_pred = tf.reduce_mean(prob, 1)  # [b]
                loss = tf.reduce_mean(-y * tf.math.log(y_pred + 1e-5) - (1 - y) * tf.math.log(1 - y_pred + 1e-5))
            else:
                t = z @ self.wt1 + self.bt1
                t = tf.nn.tanh(t)
                t = t @ self.wt2 + self.bt2
                t = tf.squeeze(t, -1)  # [b, k]
                sig_y = z @ self.wsy1 + self.bsy1
                sig_y = tf.nn.tanh(sig_y)
                sig_y = sig_y @ self.wsy2 + self.bsy2
                sig_y = tf.math.exp(sig_y) + 0.1
                sig_y = tf.squeeze(sig_y, -1)  # [b, k]
                prob = tf.clip_by_value(prob, 0.01, 0.99)
                y_preds = t + sig_y * tf.math.log((1e-5 + prob) / (1e-5 + 1 - prob))  # [b, k]
                y_pred = tf.reduce_mean(y_preds, -1)
                loss = tf.stop_gradient(mse_loss(y, y_pred))
                ll = tf.reduce_mean((0.5 * tf.square((y_preds - tf.expand_dims(y, -1)) / sig_y) + tf.math.log(sig_y)))
                ll -= tf.stop_gradient(ll)
                loss += ll
            return y_pred, loss

class GcmViModel(BaseModel):
    def __init__(self, sample_num = 32, z_dim=4, **kwargs):
        super().__init__(**kwargs)
        dense_dim, r_dim, hidden_dim = self.dense_dim, self.r_dim, self.hidden_dim
        with tf.name_scope(f'{self.name}_net'):
            self.sample_num = sample_num
            self.wr1 = get_variable([r_dim+1, hidden_dim], 'wr1')
            self.br1 = get_variable([hidden_dim], 'br1')
            self.wr2 = get_variable([hidden_dim, hidden_dim], 'wr2')
            self.br2 = get_variable([hidden_dim], 'br2')
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.b = get_variable([hidden_dim], 'b')
            self.wmu = get_variable([hidden_dim, z_dim], 'wmu')
            self.bmu = get_variable([z_dim], 'bmu')
            self.wsig = get_variable([hidden_dim, z_dim], 'wsig')
            self.bsig = get_variable([z_dim], 'bsig')
            self.wh = get_variable([z_dim, hidden_dim], 'wh')
            self.bh = get_variable([hidden_dim], 'bh')
            self.wmc = get_variable([hidden_dim, r_dim], 'wmc')
            self.bmc = get_variable([r_dim], 'bmc')
            self.wsc = get_variable([hidden_dim, r_dim], 'wsc')
            self.bsc = get_variable([r_dim], 'bsc')
            self.base = get_variable([sample_num, hidden_dim], 'base')
            if not self.is_binary:
                self.wt1 = get_variable([z_dim, hidden_dim], 'wt1')
                self.bt1 = get_variable([hidden_dim], 'bt1')
                self.wt2 = get_variable([hidden_dim, 1], 'wt2')
                self.bt2 = get_variable([1], 'bt2')
                self.wsy1 = get_variable([z_dim, hidden_dim], 'wsy1')
                self.bsy1 = get_variable([hidden_dim], 'bsy1')
                self.wsy2 = get_variable([hidden_dim, 1], 'wsy2')
                self.bsy2 = get_variable([1], 'bsy2')
    def __call__(self, inputs, y, is_test=False, **kwargs):
        x, r = inputs
        sample_num = self.sample_num
        with tf.name_scope(f'{self.name}_net'):
            h = x @ self.wx + self.b
            h = tf.nn.tanh(h)
            mu = h @ self.wmu + self.bmu  # [b, d]
            log_var = h @ self.wsig + self.bsig # [b, d]
            mu = tf.tile(tf.expand_dims(mu, 1), [1, sample_num, 1])
            sig = tf.tile(tf.expand_dims(tf.math.exp(0.5 * log_var), 1), [1, sample_num, 1])
            eps = tf.random.normal(tf.shape(mu))
            z = eps * sig + mu  # [b, k, d]  z ~ p(z|x)
            h = z @ self.wh + self.bh
            h = tf.nn.tanh(h)  # [b, k, d]
            mu_c = h @ self.wmc + self.bmc
            s_c = tf.nn.softplus(h @ self.wsc + self.bsc) + 1e-2
            s = (tf.expand_dims(r, 1) - mu_c) / s_c  # [b, k, nr]
            prob = tf.nn.sigmoid(s)
            prob = tf.exp(tf.reduce_sum(tf.math.log(prob), -1))  # [b, k]
            if self.is_binary:
                y_pred = tf.reduce_mean(prob, 1)  # [b]
                loss = tf.reduce_mean(-y * tf.math.log(y_pred + 1e-5) - (1 - y) * tf.math.log(1 - y_pred + 1e-5))
            else:
                t = z @ self.wt1 + self.bt1
                t = tf.nn.tanh(t)
                t = t @ self.wt2 + self.bt2
                t = tf.squeeze(t, -1)  # [b, k]
                sig_y = z @ self.wsy1 + self.bsy1
                sig_y = tf.nn.tanh(sig_y)
                sig_y = sig_y @ self.wsy2 + self.bsy2
                sig_y = tf.math.exp(sig_y) + 0.1
                sig_y = tf.squeeze(sig_y, -1)  # [b, k]
                prob = tf.clip_by_value(prob, 0.01, 0.99)
                y_preds = t + sig_y * tf.math.log((1e-5 + prob) / (1e-5 + 1 - prob))  # [b, k]
                y_pred = tf.reduce_mean(y_preds, -1)
                loss = tf.stop_gradient(mse_loss(y, y_pred))
            #
            _y = tf.expand_dims(y, -1)
            _ry = tf.concat([r, _y], -1)
            _ry = _ry @ self.wr1 + self.br1
            _ry = tf.nn.tanh(_ry)
            _ry = _ry @ self.wr2 + self.br2
            _ry -= tf.reduce_mean(_ry, -1, keepdims=True)
            _h = x @ self.wx + self.b + _ry
            _h = tf.nn.tanh(_h)  # [b, d]
            _mu = _h @ self.wmu + self.bmu  # [b, d]
            _log_var = _h @ self.wsig + self.bsig  # [b, d]
            _mu = tf.tile(tf.expand_dims(_mu, 1), [1, sample_num, 1])
            _sig = tf.tile(tf.expand_dims(tf.math.exp(0.5 * _log_var), 1), [1, sample_num, 1])
            _z = eps * _sig + _mu  # [b, k, d]  z ~ q(z|x,r,y)
            log_q_z = 0.5 * tf.reduce_sum(- tf.square(eps) - tf.expand_dims(_log_var, 1), -1)  # [b, k], log q(z|x), z ~ q(z|x,r,y)
            log_p_z = 0.5 * tf.reduce_sum(- tf.square((_z - mu) / sig) - tf.expand_dims(log_var, 1), -1)  # [b, k], log p(z|x), z ~ q(z|x,r,y)
            _h = _z @ self.wh + self.bh
            _h = tf.nn.tanh(_h)  # [b, k, d]
            _mu_c = _h @ self.wmc + self.bmc
            _s_c = tf.nn.softplus(_h @ self.wsc + self.bsc) + 1e-2
            _s = (tf.expand_dims(r, 1) - _mu_c) / _s_c  # [b, k, nr]
            _prob = tf.nn.sigmoid(_s)
            _prob = tf.exp(tf.reduce_sum(tf.math.log(_prob), -1))  # [b, k]  # p(r>c|x,r,z), z ~ q(z|x,r,y)
            if self.is_binary:
                log_p_y = _y * tf.math.log(_prob + 1e-5) + (1 - _y) * tf.math.log(1 - _prob + 1e-5)  # [b, k]   # log p(y|x,r,z), z ~ q(z|x,r,y)
                elb = tf.reduce_logsumexp(log_p_y + log_p_z  - log_q_z, -1)  # [b]
                elb = tf.reduce_mean(elb) - tf.math.log(float(sample_num))
                elb -= tf.stop_gradient(elb)
                loss += -elb
            else:
                _t = _z @ self.wt1 + self.bt1
                _t = tf.nn.tanh(_t)
                _t = _t @ self.wt2 + self.bt2
                _t = tf.squeeze(_t, -1)  # [b, k]
                _sig_y = _z @ self.wsy1 + self.bsy1
                _sig_y = tf.nn.tanh(_sig_y)
                _sig_y = _sig_y @ self.wsy2 + self.bsy2
                _sig_y = tf.math.exp(_sig_y) + 0.1
                _sig_y = tf.squeeze(_sig_y, -1)  # [b, k]
                _prob = tf.clip_by_value(_prob, 0.01, 0.99)
                _y_pred = _t + _sig_y * tf.math.log((_prob) / (1 - _prob))  # [b, k]
                _ll = tf.reduce_logsumexp(-(0.5 * tf.square((_y_pred - _y) / _sig_y) + tf.math.log(_sig_y)) + log_p_z  - log_q_z, -1)
                _ll = tf.reduce_mean(_ll)
                _ll -= tf.stop_gradient(_ll)
                loss += -_ll
            return y_pred, loss

class GcrmViModel(BaseModel):
    def __init__(self, sample_num = 32, z_dim=4, **kwargs):
        super().__init__(**kwargs)
        dense_dim, r_dim, hidden_dim = self.dense_dim, self.r_dim, self.hidden_dim
        with tf.name_scope(f'{self.name}_net'):
            self.sample_num = sample_num
            self.wr1 = get_variable([r_dim+1, hidden_dim], 'wr1')
            self.br1 = get_variable([hidden_dim], 'br1')
            self.wr2 = get_variable([hidden_dim, hidden_dim], 'wr2')
            self.br2 = get_variable([hidden_dim], 'br2')
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.b = get_variable([hidden_dim], 'b')
            self.wmu = get_variable([hidden_dim, z_dim], 'wmu')
            self.bmu = get_variable([z_dim], 'bmu')
            self.wsig = get_variable([hidden_dim, z_dim], 'wsig')
            self.bsig = get_variable([z_dim], 'bsig')
            self.wh = get_variable([z_dim, hidden_dim], 'wh')
            self.bh = get_variable([hidden_dim], 'bh')
            self.wmur = get_variable([hidden_dim, r_dim], 'wmur')
            self.bmur = get_variable([r_dim], 'bmur')
            self.wsigr = get_variable([hidden_dim, r_dim], 'wsigr')
            self.bsigr = get_variable([r_dim], 'bsigr')
            self.wmc = get_variable([hidden_dim, r_dim], 'wmc')
            self.bmc = get_variable([r_dim], 'bmc')
            self.wsc = get_variable([hidden_dim, r_dim], 'wsc')
            self.bsc = get_variable([r_dim], 'bsc')
            if not self.is_binary:
                self.wt1 = get_variable([z_dim, hidden_dim], 'wt1')
                self.bt1 = get_variable([hidden_dim], 'bt1')
                self.wt2 = get_variable([hidden_dim, 1], 'wt2')
                self.bt2 = get_variable([1], 'bt2')
                self.wsy1 = get_variable([z_dim, hidden_dim], 'wsy1')
                self.bsy1 = get_variable([hidden_dim], 'bsy1')
                self.wsy2 = get_variable([hidden_dim, 1], 'wsy2')
                self.bsy2 = get_variable([1], 'bsy2')
    def __call__(self, inputs, y, is_test=False, **kwargs):
        x, r = inputs
        sample_num = self.sample_num
        with tf.name_scope(f'{self.name}_net'):
            h = x @ self.wx + self.b
            h = tf.nn.tanh(h)
            mu = h @ self.wmu + self.bmu  # [b, d]
            log_var = h @ self.wsig + self.bsig # [b, d]
            mu = tf.tile(tf.expand_dims(mu, 1), [1, sample_num, 1])
            sig = tf.tile(tf.expand_dims(tf.math.exp(0.5 * log_var), 1), [1, sample_num, 1])
            eps = tf.random.normal(tf.shape(mu))
            z = eps * sig + mu  # [b, k, d]  z ~ p(z|x)
            h = z @ self.wh + self.bh
            h = tf.nn.tanh(h)  # [b, k, d]
            mu_c = h @ self.wmc + self.bmc
            s_c = tf.nn.softplus(h @ self.wsc + self.bsc) + 1e-2
            s = (tf.expand_dims(r, 1) - mu_c) / s_c  # [b, k, nr]
            prob = tf.nn.sigmoid(s)
            prob = tf.exp(tf.reduce_sum(tf.math.log(prob), -1))  # [b, k]
            if self.is_binary:
                y_pred = tf.reduce_mean(prob, 1)  # [b]
                loss = tf.reduce_mean(-y * tf.math.log(y_pred + 1e-5) - (1 - y) * tf.math.log(1 - y_pred + 1e-5))
            else:
                t = z @ self.wt1 + self.bt1
                t = tf.nn.tanh(t)
                t = t @ self.wt2 + self.bt2
                t = tf.squeeze(t, -1)  # [b, k]
                sig_y = z @ self.wsy1 + self.bsy1
                sig_y = tf.nn.tanh(sig_y)
                sig_y = sig_y @ self.wsy2 + self.bsy2
                sig_y = tf.math.exp(sig_y) + 0.1
                sig_y = tf.squeeze(sig_y, -1)  # [b, k]
                prob = tf.clip_by_value(prob, 0.01, 0.99)
                y_preds = t + sig_y * tf.math.log((1e-5 + prob) / (1e-5 + 1 - prob))  # [b, k]
                y_pred = tf.reduce_mean(y_preds, -1)
                loss = tf.stop_gradient(mse_loss(y, y_pred))
            #
            _y = tf.expand_dims(y, -1)
            _ry = tf.concat([r, _y], -1)
            _ry = _ry @ self.wr1 + self.br1
            _ry = tf.nn.tanh(_ry)
            _ry = _ry @ self.wr2 + self.br2
            _ry -= tf.reduce_mean(_ry, -1, keepdims=True)
            _h = x @ self.wx + self.b + _ry
            _h = tf.nn.tanh(_h)  # [b, d]
            _mu = _h @ self.wmu + self.bmu  # [b, d]
            _log_var = _h @ self.wsig + self.bsig  # [b, d]
            _mu = tf.tile(tf.expand_dims(_mu, 1), [1, sample_num, 1])
            _sig = tf.tile(tf.expand_dims(tf.math.exp(0.5 * _log_var), 1), [1, sample_num, 1])
            _z = eps * _sig + _mu  # [b, k, d]  z ~ q(z|x,r,y)
            log_q_z = 0.5 * tf.reduce_sum(- tf.square(eps) - tf.expand_dims(_log_var, 1), -1)  # [b, k], log q(z|x), z ~ q(z|x,r,y)
            log_p_z = 0.5 * tf.reduce_sum(- tf.square((_z - mu) / sig) - tf.expand_dims(log_var, 1), -1)  # [b, k], log p(z|x), z ~ q(z|x,r,y)
            _h = _z @ self.wh + self.bh
            _h = tf.nn.tanh(_h)  # [b, k, d]
            _mu_r = _h @ self.wmur + self.bmur  # [b, k, nr]
            _mu_r = tf.nn.sigmoid(_mu_r)
            # _log_sig_r = _h @ self.wsigr + self.bsigr  # [b, k, nr]
            # _sig_r = tf.math.exp(_log_sig_r) + 0.3
            # _log_sig_r = tf.math.log(_sig_r)
            # log_p_r = tf.reduce_sum(-0.5 * tf.square((tf.expand_dims(r, 1) - _mu_r)/_sig_r)-_log_sig_r, -1)  # [b, k]
            log_p_r = tf.reduce_sum(-0.5 * tf.square((tf.expand_dims(r, 1) - _mu_r)), -1)  # [b, k]
            _mu_c = _h @ self.wmc + self.bmc
            _s_c = tf.nn.softplus(_h @ self.wsc + self.bsc) + 1e-2
            _s = (tf.expand_dims(r, 1) - _mu_c) / _s_c  # [b, k, nr]
            _prob = tf.nn.sigmoid(_s)
            _prob = tf.exp(tf.reduce_sum(tf.math.log(_prob), -1))  # [b, k]  # p(r>c|x,r,z), z ~ q(z|x,r,y)
            if self.is_binary:
                log_p_y = _y * tf.math.log(_prob + 1e-5) + (1 - _y) * tf.math.log(1 - _prob + 1e-5)  # [b, k]   # log p(y|x,r,z), z ~ q(z|x,r,y)
                elb = tf.reduce_logsumexp(log_p_r + log_p_y + log_p_z  - log_q_z, -1)  # [b]
                elb = tf.reduce_mean(elb) - tf.math.log(float(sample_num))
                elb -= tf.stop_gradient(elb)
                loss += -elb
            else:
                _t = _z @ self.wt1 + self.bt1
                _t = tf.nn.tanh(_t)
                _t = _t @ self.wt2 + self.bt2
                _t = tf.squeeze(_t, -1)  # [b, k]
                _sig_y = _z @ self.wsy1 + self.bsy1
                _sig_y = tf.nn.tanh(_sig_y)
                _sig_y = _sig_y @ self.wsy2 + self.bsy2
                _sig_y = tf.math.exp(_sig_y) + 0.1
                _sig_y = tf.squeeze(_sig_y, -1)  # [b, k]
                _prob = tf.clip_by_value(_prob, 0.01, 0.99)
                _y_pred = _t + _sig_y * tf.math.log((_prob) / (1 - _prob))  # [b, k]
                _ll = tf.reduce_logsumexp(-(0.5 * tf.square((_y_pred - _y) / _sig_y) + tf.math.log(_sig_y)) + log_p_r + log_p_z  - log_q_z, -1)
                _ll = tf.reduce_mean(_ll)
                _ll -= tf.stop_gradient(_ll)
                loss += -_ll
            return y_pred, loss

class GcmCateModel(BaseModel):
    def __init__(self, sample_num = 32, **kwargs):
        super().__init__(**kwargs)
        dense_dim, r_dim, hidden_dim = self.dense_dim, self.r_dim, self.hidden_dim
        with tf.name_scope(f'{self.name}_net'):
            self.sample_num = sample_num
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.b = get_variable([hidden_dim], 'b')
            self.wp = get_variable([hidden_dim, sample_num], 'wp')
            self.wh = get_variable([hidden_dim, hidden_dim], 'wh')
            self.bh = get_variable([hidden_dim], 'bh')
            self.wc = get_variable([hidden_dim, r_dim], 'wc')
            self.bc = get_variable([r_dim], 'bc')
            self.ws = get_variable([hidden_dim, r_dim], 'ws')
            self.bs = get_variable([r_dim], 'bs')
            self.base = get_variable([sample_num, hidden_dim], 'base')
            if not self.is_binary:
                self.wt1 = get_variable([dense_dim, hidden_dim], 'wt1')
                self.bt1 = get_variable([hidden_dim], 'bt1')
                self.wt2 = get_variable([hidden_dim, 1], 'wt2')
                self.bt2 = get_variable([1], 'bt2')
                self.wsy1 = get_variable([dense_dim, hidden_dim], 'wsy1')
                self.bsy1 = get_variable([hidden_dim], 'bsy1')
                self.wsy2 = get_variable([hidden_dim, 1], 'wsy2')
                self.bsy2 = get_variable([1], 'bsy2')
    def __call__(self, inputs, y, **kwargs):
        x, r = inputs
        with tf.name_scope(f'{self.name}_net'):
            h = x @ self.wx + self.b
            h = tf.nn.tanh(h)
            proj_prob = tf.nn.softmax(h @ self.wp, -1)
            z = tf.expand_dims(h, 1) + tf.expand_dims(self.base, 0)
            h = z @ self.wh + self.bh
            h = tf.nn.tanh(h)
            mu_c = h @ self.wc + self.bc
            s_c = tf.nn.softplus(h @ self.ws + self.bs) + 1e-2
            s = (tf.expand_dims(r, 1) - mu_c) / s_c
            prob = tf.nn.sigmoid(s)
            if self.is_binary:
                y_pred = tf.exp(tf.reduce_sum(tf.math.log(prob), -1))
                y_pred = tf.reduce_sum(y_pred * proj_prob, 1)
                loss = tf.reduce_mean(- y * tf.math.log(y_pred + 1e-5) - (1 - y) * tf.math.log(1 - y_pred + 1e-5))
            else:
                t = x @ self.wt1 + self.bt1
                t = tf.nn.tanh(t)
                t = t @ self.wt2 + self.bt2  # [b, 1]
                sig_y = x @ self.wsy1 + self.bsy1
                sig_y = tf.nn.tanh(sig_y)
                sig_y = sig_y @ self.wsy2 + self.bsy2
                sig_y = tf.reshape(sig_y, [-1, 1])
                sig_y = tf.math.exp(sig_y) # [b, 1]
                prob_c = tf.exp(tf.reduce_sum(tf.math.log(prob), -1))  # [b, k]
                prob_c = tf.reduce_sum(prob_c * proj_prob, -1, keepdims=True)  # [b, 1]
                y_pred = tf.squeeze(t + sig_y * tf.math.log((1e-5 + prob_c) / (1e-5 + 1 - prob_c)), -1)
                loss = tf.stop_gradient(mse_loss(y, y_pred))
                ll = tf.reduce_mean((0.5 * tf.square((y_pred - y) / sig_y) + tf.math.log(sig_y)))
                ll -= tf.stop_gradient(ll)
                loss += ll
            return y_pred, loss

