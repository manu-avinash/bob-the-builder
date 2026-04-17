import keras.constraints as kc


class Constraints:
    def __init__(self):
        pass

    def maxNorm(self, max_value=None, axis=None):

        return kc.MaxNorm(max_value=max_value, axis=axis)

    def minMaxNorm(self, min_value=None, max_value=None, rate=None, axis=None):

        return kc.MinMaxNorm(
            min_value=min_value, max_value=max_value, rate=rate, axis=axis
        )

    def nonNeg(self):

        return kc.NonNeg()

    def unitNorm(self, axis=None):

        return kc.UnitNorm(axis=axis)
