import keras.losses as kl


class Losses:
    def __init__(self):
        pass

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

        return kl.BinaryCrossentropy(
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
            reduction=reduction,
            name=name,
            dtype=dtype,
        )

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

        return kl.BinaryFocalCrossentropy(
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

    def categoricalCrossentropy(
        self,
        from_logits=None,
        label_smoothing=None,
        axis=None,
        reduction=None,
        name=None,
        dtype=None,
    ):

        return kl.CategoricalCrossentropy(
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
            reduction=reduction,
            name=name,
            dtype=dtype,
        )

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

        return kl.CategoricalFocalCrossentropy(
            alpha=alpha,
            gamma=gamma,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
            reduction=reduction,
            name=name,
            dtype=dtype,
        )

    def sparseCategoricalCrossentropy(
        self,
        from_logits=None,
        ignore_class=None,
        reduction=None,
        axis=None,
        name=None,
        dtype=None,
    ):

        return kl.SparseCategoricalCrossentropy(
            from_logits=from_logits,
            ignore_class=ignore_class,
            reduction=reduction,
            axis=axis,
            name=name,
            dtype=dtype,
        )

    def poisson(self, reduction=None, name=None, dtype=None):

        return kl.Poisson(reduction=reduction, name=name, dtype=dtype)

    def ctc(self, reduction=None, name=None, dtype=None):

        return kl.CTC(reduction=reduction, name=name, dtype=dtype)

    def klDivergence(self, reduction=None, name=None, dtype=None):

        return kl.KLDivergence(reduction=reduction, name=name, dtype=dtype)

    # Regression loss
    def meanSquaredError(self, reduction=None, name=None, dtype=None):

        return kl.MeanSquaredError(reduction=reduction, name=name, dtype=dtype)

    def meanAbsoluteError(self, reduction=None, name=None, dtype=None):

        return kl.MeanAbsoluteError(reduction=reduction, name=name, dtype=dtype)

    def meanAbsolutePercentageError(self, reduction=None, name=None, dtype=None):

        return kl.MeanAbsolutePercentageError(
            reduction=reduction, name=name, dtype=dtype
        )

    def meanSquaredLogarithmicError(self, reduction=None, name=None, dtype=None):

        return kl.MeanSquaredLogarithmicError(
            reduction=reduction, name=name, dtype=dtype
        )

    def cosineSimilarity(self, axis=None, reduction=None, name=None, dtype=None):

        return kl.CosineSimilarity(
            axis=axis, reduction=reduction, name=name, dtype=dtype
        )

    def huber(self, delta=None, reduction=None, name=None, dtype=None):

        return kl.Huber(delta=delta, reduction=reduction, name=name, dtype=dtype)

    def logCosh(self, reduction=None, name=None, dtype=None):

        return kl.LogCosh(reduction=reduction, name=name, dtype=dtype)

    def tversky(
        self,
        alpha=None,
        beta=None,
        reduction=None,
        name=None,
        axis=None,
        dtype=None,
    ):

        return kl.Tversky(
            alpha=alpha,
            beta=beta,
            reduction=reduction,
            name=name,
            axis=axis,
            dtype=dtype,
        )

    def dice(self, reduction=None, name=None, axis=None, dtype=None):

        return kl.Dice(reduction=reduction, name=name, axis=axis, dtype=dtype)

    # Hinge losses for "maximum-margin" classification
    def hinge(self, reduction=None, name=None, dtype=None):

        return kl.Hinge(reduction=reduction, name=name, dtype=dtype)

    def squaredHinge(self, reduction=None, name=None, dtype=None):

        return kl.SquaredHinge(reduction=reduction, name=name, dtype=dtype)

    def categoricalHinge(self, reduction=None, name=None, dtype=None):

        return kl.CategoricalHinge(reduction=reduction, name=name, dtype=dtype)

    def categoricalGeneralizedCrossEntropy(
        self,
        q=None,
        reduction=None,
        name=None,
        dtype=None,
    ):

        return kl.CategoricalGeneralizedCrossEntropy(
            q=q, reduction=reduction, name=name, dtype=dtype
        )

    def circle(
        self,
        gamma=None,
        margin=None,
        remove_diagonal=None,
        reduction=None,
        name=None,
        dtype=None,
    ):

        return kl.Circle(
            gamma=gamma,
            margin=margin,
            remove_diagonal=remove_diagonal,
            reduction=reduction,
            name=name,
            dtype=dtype,
        )
