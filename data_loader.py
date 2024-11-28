import csv
import numpy as np
from ucimlrepo import fetch_ucirepo

class DataLoader(object):
    _cache = {}
    def load(self, name, seed):
        if name == 'Adult':
            if name not in self._cache:
                adult = fetch_ucirepo(id=2)
                self._cache[name] = adult
            else:
                adult = self._cache[name]
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
                xx = [[0.0] * l for _ in x]
                for i, j in zip(x, xx):
                    j[i] = 1.0
                xs.append(xx)
            r = [list(i) for i in zip(*rs)]
            x = [sum(i, []) for i in zip(*xs)]
            y = [1.0 if i == '>50K.' else 0 for i in y['income'].tolist()]
            train_num = int(num * 0.8)
            rnd_id = list(range(num))
            np.random.seed(seed)
            np.random.shuffle(rnd_id)
            x_train = np.float32(np.array([x[i] for i in rnd_id[:train_num]]))
            r_train = np.float32(np.array([r[i] for i in rnd_id[:train_num]]))
            y_train = np.float32(np.array([y[i] for i in rnd_id[:train_num]]))
            x_test = np.float32(np.array([x[i] for i in rnd_id[train_num:]]))
            r_test = np.float32(np.array([r[i] for i in rnd_id[train_num:]]))
            y_test = np.float32(np.array([y[i] for i in rnd_id[train_num:]]))
            return x_train, r_train, y_train, x_test, r_test, y_test, 33, 4, 1
        elif name == 'Diabetes':
            if name not in self._cache:
                diabetes = fetch_ucirepo(id=891)
                self._cache[name] = diabetes
            else:
                diabetes = self._cache[name]
            X = diabetes.data.features
            y = diabetes.data.targets
            xs, rs = [], []
            num = 253680
            for n in X.columns:
                if n in ['BMI', 'HvyAlcoholConsump', 'Smoker', 'Age']:
                    tmp = X[n].tolist()
                    a, b = min(tmp), max(tmp)
                    tmp = [(float(i) - a) / (b - a) for i in tmp]
                    rs.append(tmp)
                elif n in ['HighBP', 'HighChol', 'CholCheck', 'Stroke',
                           'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                           'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                           'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Education',
                           'Income']:
                    tmp = X[n].tolist()
                    k = set(tmp)
                    k = list(k)
                    l = len(k)
                    x = [k.index(i) for i in tmp]
                    xx = [[0.0] * l for i in x]
                    for i, j in zip(x, xx):
                        j[i] = 1.0
                    xs.append(xx)
            r = [list(i) for i in zip(*rs)]  # [b, 4]
            x = [sum(i, []) for i in zip(*xs)]  # [b, 105]
            y = y['Diabetes_binary'].tolist()
            train_num = int(num * 0.8)
            rnd_id = list(range(num))
            np.random.seed(seed)
            np.random.shuffle(rnd_id)
            x_train = np.float32(np.array([x[i] for i in rnd_id[:train_num]]))
            r_train = np.float32(np.array([r[i] for i in rnd_id[:train_num]]))
            y_train = np.float32(np.array([y[i] for i in rnd_id[:train_num]]))
            x_test = np.float32(np.array([x[i] for i in rnd_id[train_num:]]))
            r_test = np.float32(np.array([r[i] for i in rnd_id[train_num:]]))
            y_test = np.float32(np.array([y[i] for i in rnd_id[train_num:]]))
            return x_train, r_train, y_train, x_test, r_test, y_test, 105, 4, 1
        elif name == 'BlogFeedback':
            import zipfile
            with zipfile.ZipFile('resource/blogfeedback.zip', 'r') as zip_ref:
                zip_ref.extractall('resource/blog')
            with open('resource/blog/blogData_train.csv', newline='') as f:
                blog = [i for i in csv.reader(f)]
            X = [[float(j) for j in i[:280]] for i in blog]
            b = [min(i) for i in zip(*X)]
            X = [[j - k for j, k in zip(i, b)] for i in X]
            X = [[np.log(1 + j) for j in i] for i in X]
            a = [max(i) for i in zip(*X)]
            b = [min(i) for i in zip(*X)]
            X = [[(i[j] - b[j]) / (a[j] - b[j] + 1e-5) for j in range(280)] for i in X]
            y = [float(i[280]) for i in blog]
            y = [np.log(1+j) for j in y]
            a = max(y)
            y = [i / a for i in y]
            num = 52397
            train_num = int(num * 0.8)
            rnd_id = list(range(num))
            np.random.seed(seed)
            np.random.shuffle(rnd_id)
            r = [i[50:55] + i[277:280] for i in X]
            x = [i[0:50] + i[55:277] for i in X]
            x_train = np.float32(np.array([x[i] for i in rnd_id[:train_num]]))
            r_train = np.float32(np.array([r[i] for i in rnd_id[:train_num]]))
            y_train = np.float32(np.array([y[i] for i in rnd_id[:train_num]]))
            x_test = np.float32(np.array([x[i] for i in rnd_id[train_num:]]))
            r_test = np.float32(np.array([r[i] for i in rnd_id[train_num:]]))
            y_test = np.float32(np.array([y[i] for i in rnd_id[train_num:]]))
            return x_train, r_train, y_train, x_test, r_test, y_test, 272, 8, 0
