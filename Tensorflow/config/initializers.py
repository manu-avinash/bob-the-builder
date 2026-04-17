import keras.initializers as init


class Initializer:
    def __init__(self):
        pass

    # Kernel_initializer
    def randomNormal(self, mean=None, stddev=None, seed=None):

        return init.RandomNormal(mean=mean, stddev=stddev, seed=seed)

    def randomUniform(self, minval=None, maxval=None, seed=None):

        return init.RandomUniform(minval=minval, maxval=maxval, seed=seed)

    def truncatedNormal(self, mean=None, stddev=None, seed=None):

        return init.TruncatedNormal(mean=mean, stddev=stddev, seed=seed)

    def zeros(self):

        return init.Zeros()

    def ones(self):

        return init.Ones()

    def glorotNormal(self, seed=None):

        return init.GlorotNormal(seed=seed)

    def glorotUniform(self, seed=None):

        return init.GlorotUniform(seed=seed)

    def heNormal(self, seed=None):

        return init.HeNormal(seed=seed)

    def heUniform(self, seed=None):

        return init.HeUniform(seed=seed)

    def orthogonal(self, gain=None, seed=None):

        return init.Orthogonal(gain=gain, seed=seed)

    def constant(self, value=None):

        return init.Constant(value=value)

    def varianceScaling(self, scale=None, mode=None, distribution=None, seed=None):

        return init.VarianceScaling(
            scale=scale, mode=mode, distribution=distribution, seed=seed
        )

    def lecunNormal(self, seed=None):

        return init.LecunNormal(seed=seed)

    def lecumUniform(self, seed=None):

        return init.LecunUniform(seed=seed)

    def identity(self, gain=None):

        return init.Identity(gain=gain)
