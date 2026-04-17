import keras.optimizers as ko


class Optimizers:
    def __init__(self):
        self.built = {}

    def SGD(
        self,
        learning_rate=None,
        momentum=None,
        nesterov=None,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=None,
        ema_momentum=None,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name=None,
        **kwargs
    ):

        return ko.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            name=name,
            **kwargs
        )

    def RMSprop(
        self,
        learning_rate=None,
        rho=None,
        momentum=None,
        epsilon=None,
        centered=None,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=None,
        ema_momentum=None,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name=None,
        **kwargs
    ):

        return ko.RMSprop(
            learning_rate=learning_rate,
            rho=rho,
            momentum=momentum,
            epsilon=epsilon,
            centered=centered,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            name=name,
            **kwargs
        )

    def Adam(
        self,
        learning_rate=None,
        beta_1=None,
        beta_2=None,
        epsilon=None,
        amsgrad=None,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=None,
        ema_momentum=None,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name=None,
        **kwargs
    ):

        return ko.Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            name=name,
            **kwargs
        )

    def AdamW(
        self,
        learning_rate=None,
        weight_decay=None,
        beta_1=None,
        beta_2=None,
        epsilon=None,
        amsgrad=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=None,
        ema_momentum=None,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name=None,
        **kwargs
    ):

        return ko.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            name=name,
            **kwargs
        )

    def Adadelta(
        self,
        learning_rate=None,
        rho=None,
        epsilon=None,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=None,
        ema_momentum=None,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name=None,
        **kwargs
    ):

        return ko.Adadelta(
            learning_rate=learning_rate,
            rho=rho,
            epsilon=epsilon,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            name=name,
            **kwargs
        )

    def Adagrad(
        self,
        learning_rate=None,
        initial_accumulator_value=None,
        epsilon=None,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=None,
        ema_momentum=None,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name=None,
        **kwargs
    ):

        return ko.Adagrad(
            learning_rate=learning_rate,
            initial_accumulator_value=initial_accumulator_value,
            epsilon=epsilon,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            name=name,
            **kwargs
        )

    def Adamax(
        self,
        learning_rate=None,
        beta_1=None,
        beta_2=None,
        epsilon=None,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=None,
        ema_momentum=None,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name=None,
        **kwargs
    ):

        return ko.Adamax(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            name=name,
            **kwargs
        )

    def Adafactor(
        self,
        learning_rate=None,
        beta_2_decay=None,
        epsilon_1=None,
        epsilon_2=None,
        clip_threshold=None,
        relative_step=None,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=None,
        ema_momentum=None,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name=None,
        **kwargs
    ):

        return ko.Adafactor(
            learning_rate=learning_rate,
            beta_2_decay=beta_2_decay,
            epsilon_1=epsilon_1,
            epsilon_2=epsilon_2,
            clip_threshold=clip_threshold,
            relative_step=relative_step,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            name=name,
            **kwargs
        )

    def Nadam(
        self,
        learning_rate=None,
        beta_1=None,
        beta_2=None,
        epsilon=None,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=None,
        ema_momentum=None,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name=None,
        **kwargs
    ):

        return ko.Nadam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            name=name,
            **kwargs
        )

    def Ftrl(
        self,
        learning_rate=None,
        learning_rate_power=None,
        initial_accumulator_value=None,
        l1_regularization_strength=None,
        l2_regularization_strength=None,
        l2_shrinkage_regularization_strength=None,
        beta=None,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=None,
        ema_momentum=None,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name=None,
        **kwargs
    ):

        return ko.Ftrl(
            learning_rate=learning_rate,
            learning_rate_power=learning_rate_power,
            initial_accumulator_value=initial_accumulator_value,
            l1_regularization_strength=l1_regularization_strength,
            l2_regularization_strength=l2_regularization_strength,
            l2_shrinkage_regularization_strength=l2_shrinkage_regularization_strength,
            beta=beta,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            name=name,
            **kwargs
        )

    def Lion(
        self,
        learning_rate=None,
        beta_1=None,
        beta_2=None,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=None,
        ema_momentum=None,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name=None,
        **kwargs
    ):

        return ko.Lion(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            name=name,
            **kwargs
        )

    def Lamb(
        self,
        learning_rate=None,
        beta_1=None,
        beta_2=None,
        epsilon=None,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=None,
        ema_momentum=None,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name=None,
        **kwargs
    ):

        return ko.Lamb(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            name=name,
            **kwargs
        )

    def Loss_Scale_Optimizer(
        self,
        inner_optimizer=None,
        initial_scale=None,
        dynamic_growth_steps=None,
        name=None,
        **kwargs
    ):

        return ko.LossScaleOptimizer(
            inner_optimizer=inner_optimizer,
            initial_scale=initial_scale,
            dynamic_growth_steps=dynamic_growth_steps,
            name=name,
            **kwargs
        )

    def Muon(
        self,
        learning_rate=None,
        adam_beta_1=None,
        adam_beta_2=None,
        adam_weight_decay=None,
        epsilon=None,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=None,
        ema_momentum=None,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name=None,
        exclude_layers=None,
        exclude_embeddings=None,
        muon_a=None,
        muon_b=None,
        muon_c=None,
        adam_lr_ratio=None,
        momentum=None,
        ns_steps=None,
        nesterov=None,
        rms_rate=None,
        **kwargs
    ):

        return ko.Muon(
            learning_rate=learning_rate,
            adam_beta_1=adam_beta_1,
            adam_beta_2=adam_beta_2,
            adam_weight_decay=adam_weight_decay,
            epsilon=epsilon,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            name=name,
            exclude_layers=exclude_layers,
            exclude_embeddings=exclude_embeddings,
            muon_a=muon_a,
            muon_b=muon_b,
            muon_c=muon_c,
            adam_lr_ratio=adam_lr_ratio,
            momentum=momentum,
            ns_steps=ns_steps,
            nesterov=nesterov,
            rms_rate=rms_rate,
            **kwargs
        )
