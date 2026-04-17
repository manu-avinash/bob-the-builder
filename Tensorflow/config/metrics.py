import keras.metrics as km


class Metrics:
    def __init__(self):
        self.built = {"metrics": []}

    # Accuracy metrics
    def Accuracy(self, name=None, dtype=None):
        layer = km.Accuracy(name=name, dtype=dtype)
        self.built["metrics"].append(layer)

    def BinaryAccuracy(self, name=None, dtype=None, threshold=None):
        layer = km.BinaryAccuracy(name=name, dtype=dtype, threshold=threshold)
        self.built["metrics"].append(layer)

    def CategoricalAccuracy(self, name=None, dtype=None):
        layer = km.CategoricalAccuracy(name=name, dtype=dtype)
        self.built["metrics"].append(layer)

    def SparseCategoricalAccuracy(self, name=None, dtype=None):
        layer = km.SparseCategoricalAccuracy(name=name, dtype=dtype)
        self.built["metrics"].append(layer)

    def TopKCategoricalAccuracy(self, k=None, name=None, dtype=None):
        layer = km.TopKCategoricalAccuracy(k=k, name=name, dtype=dtype)
        self.built["metrics"].append(layer)

    def SparseTopKCategoricalAccuracy(
        self, k=None, name=None, dtype=None, from_sorted_ids=None
    ):
        layer = km.SparseTopKCategoricalAccuracy(
            k=k, name=name, dtype=dtype, from_sorted_ids=from_sorted_ids
        )
        self.built["metrics"].append(layer)

    # Probabilistic metrics
    def BinaryCrossentropy(
        self, name=None, dtype=None, from_logits=None, label_smoothing=None
    ):
        layer = km.BinaryCrossentropy(
            name=name,
            dtype=dtype,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
        )
        self.built["metrics"].append(layer)

    def CategoricalCrossentropy(
        self, name=None, dtype=None, from_logits=None, label_smoothing=None, axis=None
    ):
        layer = km.CategoricalCrossentropy(
            name=name,
            dtype=dtype,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
        )
        self.built["metrics"].append(layer)

    def SparseCategoricalCrossentropy(
        self, name=None, dtype=None, from_logits=None, ignore_class=None, axis=None
    ):
        layer = km.SparseCategoricalCrossentropy(
            name=name,
            dtype=dtype,
            from_logits=from_logits,
            ignore_class=ignore_class,
            axis=axis,
        )
        self.built["metrics"].append(layer)

    def KLDivergence(self, name=None, dtype=None):
        layer = km.KLDivergence(name=name, dtype=dtype)
        self.built["metrics"].append(layer)

    def Poisson(self, name=None, dtype=None):
        layer = km.Poisson(name=name, dtype=dtype)
        self.built["metrics"].append(layer)

    # Regression metrics
    def MeanSquaredError(self, name=None, dtype=None):
        layer = km.MeanSquaredError(name=name, dtype=dtype)
        self.built["metrics"].append(layer)

    def RootMeanSquaredError(self, name=None, dtype=None):
        layer = km.RootMeanSquaredError(name=name, dtype=dtype)
        self.built["metrics"].append(layer)

    def MeanAbsoluteError(self, name=None, dtype=None):
        layer = km.MeanAbsoluteError(name=name, dtype=dtype)
        self.built["metrics"].append(layer)

    def MeanAbsolutePercentageError(self, name=None, dtype=None):
        layer = km.MeanAbsolutePercentageError(name=name, dtype=dtype)
        self.built["metrics"].append(layer)

    def MeanSquaredLogarithmicError(self, name=None, dtype=None):
        layer = km.MeanSquaredLogarithmicError(name=name, dtype=dtype)
        self.built["metrics"].append(layer)

    def CosineSimilarity(self, name=None, dtype=None, axis=None):
        layer = km.CosineSimilarity(name=name, dtype=dtype, axis=axis)
        self.built["metrics"].append(layer)

    def LogCoshError(self, name=None, dtype=None):
        layer = km.LogCoshError(name=name, dtype=dtype)
        self.built["metrics"].append(layer)

    def R2Score(
        self, class_aggregation=None, num_regressors=None, name=None, dtype=None
    ):
        layer = km.R2Score(
            class_aggregation=class_aggregation,
            num_regressors=num_regressors,
            name=name,
            dtype=dtype,
        )
        self.built["metrics"].append(layer)

    # Classification metrics based on True/False positives & negatives
    def AUC(
        self,
        num_thresholds=None,
        curve=None,
        summation_method=None,
        name=None,
        dtype=None,
        thresholds=None,
        multi_label=None,
        num_labels=None,
        label_weights=None,
        from_logits=None,
    ):
        layer = km.AUC(
            num_thresholds=num_thresholds,
            curve=curve,
            summation_method=summation_method,
            name=name,
            dtype=dtype,
            thresholds=thresholds,
            multi_label=multi_label,
            num_labels=num_labels,
            label_weights=label_weights,
            from_logits=from_logits,
        )
        self.built["metrics"].append(layer)

    def Precision(
        self, thresholds=None, top_k=None, class_id=None, name=None, dtype=None
    ):
        layer = km.Precision(
            thresholds=thresholds,
            top_k=top_k,
            class_id=class_id,
            name=name,
            dtype=dtype,
        )
        self.built["metrics"].append(layer)

    def Recall(self, thresholds=None, top_k=None, class_id=None, name=None, dtype=None):
        layer = km.Recall(
            thresholds=thresholds,
            top_k=top_k,
            class_id=class_id,
            name=name,
            dtype=dtype,
        )
        self.built["metrics"].append(layer)

    def TruePositives(self, thresholds=None, name=None, dtype=None):
        layer = km.TruePositives(thresholds=thresholds, name=name, dtype=dtype)
        self.built["metrics"].append(layer)

    def TrueNegatives(self, thresholds=None, name=None, dtype=None):
        layer = km.TrueNegatives(thresholds=thresholds, name=name, dtype=dtype)
        self.built["metrics"].append(layer)

    def FalsePositives(self, thresholds=None, name=None, dtype=None):
        layer = km.FalsePositives(thresholds=thresholds, name=name, dtype=dtype)
        self.built["metrics"].append(layer)

    def FalseNegatives(self, thresholds=None, name=None, dtype=None):
        layer = km.FalseNegatives(thresholds=thresholds, name=name, dtype=dtype)
        self.built["metrics"].append(layer)

    def PrecisionAtRecall(
        self, recall=None, num_thresholds=None, class_id=None, name=None, dtype=None
    ):
        layer = km.PrecisionAtRecall(
            recall=recall,
            num_thresholds=num_thresholds,
            class_id=class_id,
            name=name,
            dtype=dtype,
        )
        self.built["metrics"].append(layer)

    def RecallAtPrecision(
        self, precision=None, num_thresholds=None, class_id=None, name=None, dtype=None
    ):
        layer = km.RecallAtPrecision(
            precision=precision,
            num_thresholds=num_thresholds,
            class_id=class_id,
            name=name,
            dtype=dtype,
        )
        self.built["metrics"].append(layer)

    def SensitivityAtSpecificity(
        self,
        specificity=None,
        num_thresholds=None,
        class_id=None,
        name=None,
        dtype=None,
    ):
        layer = km.SensitivityAtSpecificity(
            specificity=specificity,
            num_thresholds=num_thresholds,
            class_id=class_id,
            name=name,
            dtype=dtype,
        )
        self.built["metrics"].append(layer)

    def SpecificityAtSensitivity(
        self,
        sensitivity=None,
        num_thresholds=None,
        class_id=None,
        name=None,
        dtype=None,
    ):
        layer = km.SpecificityAtSensitivity(
            sensitivity=sensitivity,
            num_thresholds=num_thresholds,
            class_id=class_id,
            name=name,
            dtype=dtype,
        )
        self.built["metrics"].append(layer)

    def F1Score(self, average=None, threshold=None, name=None, dtype=None):
        layer = km.F1Score(average=average, threshold=threshold, name=name, dtype=dtype)
        self.built["metrics"].append(layer)

    def FBetaScore(
        self, average=None, beta=None, threshold=None, name=None, dtype=None
    ):
        layer = km.FBetaScore(
            average=average, beta=beta, threshold=threshold, name=name, dtype=dtype
        )
        self.built["metrics"].append(layer)

    def PearsonCorrelation(self, name=None, dtype=None, axis=None):
        layer = km.PearsonCorrelation(name=name, dtype=dtype, axis=axis)
        self.built["metrics"].append(layer)

    def ConcordanceCorrelation(self, name=None, dtype=None, axis=None):
        layer = km.ConcordanceCorrelation(name=name, dtype=dtype, axis=axis)
        self.built["metrics"].append(layer)

    # Image segmentation metrics
    def IoU(
        self,
        num_classes=None,
        target_class_ids=None,
        name=None,
        dtype=None,
        ignore_class=None,
        sparse_y_true=None,
        sparse_y_pred=None,
        axis=None,
    ):
        layer = km.IoU(
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            name=name,
            dtype=dtype,
            ignore_class=ignore_class,
            sparse_y_true=sparse_y_true,
            sparse_y_pred=sparse_y_pred,
            axis=axis,
        )
        self.built["metrics"].append(layer)

    def BinaryIoU(self, target_class_ids=None, threshold=None, name=None, dtype=None):
        layer = km.BinaryIoU(
            target_class_ids=target_class_ids,
            threshold=threshold,
            name=name,
            dtype=dtype,
        )
        self.built["metrics"].append(layer)

    def OneHotIoU(
        self,
        num_classes=None,
        target_class_ids=None,
        name=None,
        dtype=None,
        ignore_class=None,
        sparse_y_pred=None,
        axis=None,
    ):
        layer = km.OneHotIoU(
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            name=name,
            dtype=dtype,
            ignore_class=ignore_class,
            sparse_y_pred=sparse_y_pred,
            axis=axis,
        )
        self.built["metrics"].append(layer)

    def OneHotMeanIoU(
        self,
        num_classes=None,
        name=None,
        dtype=None,
        ignore_class=None,
        sparse_y_pred=None,
        axis=None,
    ):
        layer = km.OneHotMeanIoU(
            num_classes=num_classes,
            name=name,
            dtype=dtype,
            ignore_class=ignore_class,
            sparse_y_pred=sparse_y_pred,
            axis=axis,
        )
        self.built["metrics"].append(layer)

    def MeanIoU(
        self,
        num_classes=None,
        name=None,
        dtype=None,
        ignore_class=None,
        sparse_y_true=None,
        sparse_y_pred=None,
        axis=None,
    ):
        layer = km.MeanIoU(
            num_classes=num_classes,
            name=name,
            dtype=dtype,
            ignore_class=ignore_class,
            sparse_y_true=sparse_y_true,
            sparse_y_pred=sparse_y_pred,
            axis=axis,
        )
        self.built["metrics"].append(layer)

    # Hinge metrics for "maximum-margin" classification
    def Hinge(self, name=None, dtype=None):
        layer = km.Hinge(name=name, dtype=dtype)
        self.built["metrics"].append(layer)

    def SquaredHinge(self, name=None, dtype=None):
        layer = km.SquaredHinge(name=name, dtype=dtype)
        self.built["metrics"].append(layer)

    def CategoricalHinge(self, name=None, dtype=None):
        layer = km.CategoricalHinge(name=name, dtype=dtype)
        self.built["metrics"].append(layer)

    # Metric wrappers and reduction metrics
    def MeanMetricWrapper(self, fn=None, name=None, dtype=None, **kwargs):
        layer = km.MeanMetricWrapper(fn=fn, name=name, dtype=dtype, **kwargs)
        self.built["metrics"].append(layer)

    def Mean(self, name=None, dtype=None):
        layer = km.Mean(name=name, dtype=dtype)
        self.built["metrics"].append(layer)

    def Sum(self, name=None, dtype=None):
        layer = km.Sum(name=name, dtype=dtype)
        self.built["metrics"].append(layer)
