import keras.initializers as init


class Initializer:
    def __init__(self):
        self.built = {}

    # Kernel_initializer
    def randomNormal(self, mean=None, stddev=None, seed=None):
        layer = init.RandomNormal(mean=mean, stddev=stddev, seed=seed)
        self.built[layer.name] = layer
        return layer

    def randomUniform(self, minval=None, maxval=None, seed=None):
        layer = init.RandomUniform(minval=minval, maxval=maxval, seed=seed)
        self.built[layer.name] = layer
        return layer

    def truncatedNormal(self, mean=None, stddev=None, seed=None):
        layer = init.TruncatedNormal(mean=mean, stddev=stddev, seed=seed)
        self.built[layer.name] = layer
        return layer

    def zeros(self):
        layer = init.Zeros()
        self.built[layer.name] = layer
        return layer

    def ones(self):
        layer = init.Ones()
        self.built[layer.name] = layer
        return layer

    def glorotNormal(self, seed=None):
        layer = init.GlorotNormal(seed=seed)
        self.built[layer.name] = layer
        return layer

    def glorotUniform(self, seed=None):
        layer = init.GlorotUniform(seed=seed)
        self.built[layer.name] = layer
        return layer

    def heNormal(self, seed=None):
        layer = init.HeNormal(seed=seed)
        self.built[layer.name] = layer
        return layer

    def heUniform(self, seed=None):
        layer = init.HeUniform(seed=seed)
        self.built[layer.name] = layer
        return layer

    def orthogonal(self, gain=None, seed=None):
        layer = init.Orthogonal(gain=gain, seed=seed)
        self.built[layer.name] = layer
        return layer

    def constant(self, value=None):
        layer = init.Constant(value=value)
        self.built[layer.name] = layer
        return layer

    def varianceScaling(self, scale=None, mode=None, distribution=None, seed=None):
        layer = init.VarianceScaling(
            scale=scale, mode=mode, distribution=distribution, seed=seed
        )
        self.built[layer.name] = layer
        return layer

    def lecunNormal(self, seed=None):
        layer = init.LecunNormal(seed=seed)
        self.built[layer.name] = layer
        return layer

    def lecumUniform(self, seed=None):
        layer = init.LecunUniform(seed=seed)
        self.built[layer.name] = layer
        return layer

    def identity(self, gain=None):
        layer = init.Identity(gain=gain)
        self.built[layer.name] = layer
        return layer
