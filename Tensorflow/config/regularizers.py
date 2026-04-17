import keras.regularizers as kr


class Regularizer:
    def __init__(self):
        pass

    def l1(self, l1=None):

        return kr.L1(l1=l1)

    def l2(self, l2=None):

        return kr.L2(l2=l2)

    def l1l2(self, l1=None, l2=None):

        return kr.L1L2(l1=l1, l2=l2)

    def orthogonalRegularizer(self, factor=None, mode=None):

        return kr.OrthogonalRegularizer(factor=factor, mode=mode)
