import keras.losses as kl


class Losses:
    def __init__(self):
        self.built = {}

    # Probabilistic losses
    def binaryCrossentropy(
        self,
        from_logits=None,
        label_smoothing=None,
        axis=None,
        reduction=None,
        name=None,
        dtype=None,
    ):
        layer = kl.BinaryCrossentropy(
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
            reduction=reduction,
            name=name,
            dtype=dtype,
        )
        self.built[layer.name] = layer
        return layer

    def binaryFocalCrossentropy(
        self,
        apply_class_balancing=None,
        alpha=None,
        gamma=None,
        from_logits=None,
        label_smoothing=None,
        axis=None,
        reduction=None,
        name=None,
        dtype=None,
    ):
        layer = kl.BinaryFocalCrossentropy(
            apply_class_balancing=apply_class_balancing,
            alpha=alpha,
            gamma=gamma,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
            reduction=reduction,
            name=name,
            dtype=dtype,
        )
        self.built[layer.name] = layer
        return layer

    def categoricalCrossentropy(
        self,
        from_logits=None,
        label_smoothing=None,
        axis=None,
        reduction=None,
        name=None,
        dtype=None,
    ):
        layer = kl.CategoricalCrossentropy(
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
            reduction=reduction,
            name=name,
            dtype=dtype,
        )
        self.built[layer.name] = layer
        return layer

    def categoricalFocalCrossentropy(
        self,
        alpha=None,
        gamma=None,
        from_logits=None,
        label_smoothing=None,
        axis=None,
        reduction=None,
        name=None,
        dtype=None,
    ):
        layer = kl.CategoricalFocalCrossentropy(
            alpha=alpha,
            gamma=gamma,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
            reduction=reduction,
            name=name,
            dtype=dtype,
        )
        self.built[layer.name] = layer
        return layer

    def sparseCategoricalCrossentropy(
        self,
        from_logits=None,
        ignore_class=None,
        reduction=None,
        axis=None,
        name=None,
        dtype=None,
    ):
        layer = kl.SparseCategoricalCrossentropy(
            from_logits=from_logits,
            ignore_class=ignore_class,
            reduction=reduction,
            axis=axis,
            name=name,
            dtype=dtype,
        )
        self.built[layer.name] = layer
        return layer

    def poisson(self, reduction=None, name=None, dtype=None):
        layer = kl.Poisson(reduction=reduction, name=name, dtype=dtype)
        self.built[layer.name] = layer
        return layer

    def ctc(self, reduction=None, name=None, dtype=None):
        layer = kl.CTC(reduction=reduction, name=name, dtype=dtype)
        self.built[layer.name] = layer
        return layer

    def klDivergence(self, reduction=None, name=None, dtype=None):
        layer = kl.KLDivergence(reduction=reduction, name=name, dtype=dtype)
        self.built[layer.name] = layer
        return layer

    # Regression loss
    def meanSquaredError(self, reduction=None, name=None, dtype=None):
        layer = kl.MeanSquaredError(reduction=reduction, name=name, dtype=dtype)
        self.built[layer.name] = layer
        return layer

    def meanAbsoluteError(self, reduction=None, name=None, dtype=None):
        layer = kl.MeanAbsoluteError(reduction=reduction, name=name, dtype=dtype)
        self.built[layer.name] = layer
        return layer

    def meanAbsolutePercentageError(self, reduction=None, name=None, dtype=None):
        layer = kl.MeanAbsolutePercentageError(
            reduction=reduction, name=name, dtype=dtype
        )
        self.built[layer.name] = layer
        return layer

    def meanSquaredLogarithmicError(self, reduction=None, name=None, dtype=None):
        layer = kl.MeanSquaredLogarithmicError(
            reduction=reduction, name=name, dtype=dtype
        )
        self.built[layer.name] = layer
        return layer

    def cosineSimilarity(self, axis=None, reduction=None, name=None, dtype=None):
        layer = kl.CosineSimilarity(
            axis=axis, reduction=reduction, name=name, dtype=dtype
        )
        self.built[layer.name] = layer
        return layer

    def huber(self, delta=None, reduction=None, name=None, dtype=None):
        layer = kl.Huber(delta=delta, reduction=reduction, name=name, dtype=dtype)
        self.built[layer.name] = layer
        return layer

    def logCosh(self, reduction=None, name=None, dtype=None):
        layer = kl.LogCosh(reduction=reduction, name=name, dtype=dtype)
        self.built[layer.name] = layer
        return layer

    def tversky(
        self,
        alpha=None,
        beta=None,
        reduction=None,
        name=None,
        axis=None,
        dtype=None,
    ):
        layer = kl.Tversky(
            alpha=alpha,
            beta=beta,
            reduction=reduction,
            name=name,
            axis=axis,
            dtype=dtype,
        )
        self.built[layer.name] = layer
        return layer

    def dice(self, reduction=None, name=None, axis=None, dtype=None):
        layer = kl.Dice(reduction=reduction, name=name, axis=axis, dtype=dtype)
        self.built[layer.name] = layer
        return layer

    # Hinge losses for "maximum-margin" classification
    def hinge(self, reduction=None, name=None, dtype=None):
        layer = kl.Hinge(reduction=reduction, name=name, dtype=dtype)
        self.built[layer.name] = layer
        return layer

    def squaredHinge(self, reduction=None, name=None, dtype=None):
        layer = kl.SquaredHinge(reduction=reduction, name=name, dtype=dtype)
        self.built[layer.name] = layer
        return layer

    def categoricalHinge(self, reduction=None, name=None, dtype=None):
        layer = kl.CategoricalHinge(reduction=reduction, name=name, dtype=dtype)
        self.built[layer.name] = layer
        return layer

    def categoricalGeneralizedCrossEntropy(
        self,
        q=None,
        reduction=None,
        name=None,
        dtype=None,
    ):
        layer = kl.CategoricalGeneralizedCrossEntropy(
            q=q, reduction=reduction, name=name, dtype=dtype
        )
        self.built[layer.name] = layer
        return layer

    def circle(
        self,
        gamma=None,
        margin=None,
        remove_diagonal=None,
        reduction=None,
        name=None,
        dtype=None,
    ):
        layer = kl.Circle(
            gamma=gamma,
            margin=margin,
            remove_diagonal=remove_diagonal,
            reduction=reduction,
            name=name,
            dtype=dtype,
        )
        self.built[layer.name] = layer
        return layer
