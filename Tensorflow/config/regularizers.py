import keras.regularizers as kr


class Regularizer:
    def __init__(self):
        self.built = {}

    def l1(self, l1=None):
        layer = kr.L1(l1=l1)
        self.built[layer.name] = layer
        return layer

    def l2(self, l2=None):
        layer = kr.L2(l2=l2)
        self.built[layer.name] = layer
        return layer

    def l1l2(self, l1=None, l2=None):
        layer = kr.L1L2(l1=l1, l2=l2)
        self.built[layer.name] = layer
        return layer

    def orthogonalRegularizer(self, factor=None, mode=None):
        layer = kr.OrthogonalRegularizer(factor=factor, mode=mode)
        self.built[layer.name] = layer
        return layer
