import keras.constraints as kc


class Constraints:
    def __init__(self):
        self.built = {}

    def maxNorm(self, max_value=None, axis=None):
        layer = kc.MaxNorm(max_value=max_value, axis=axis)
        self.built[layer.name] = layer
        return layer

    def minMaxNorm(self, min_value=None, max_value=None, rate=None, axis=None):
        layer = kc.MinMaxNorm(
            min_value=min_value, max_value=max_value, rate=rate, axis=axis
        )
        self.built[layer.name] = layer
        return layer

    def nonNeg(self):
        layer = kc.NonNeg()
        self.built[layer.name] = layer
        return layer

    def unitNorm(self, axis=None):
        layer = kc.UnitNorm(axis=axis)
        self.built[layer.name] = layer
        return layer
