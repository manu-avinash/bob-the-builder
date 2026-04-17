import keras.optimizers.schedules as kos


class Scheduler:
    def __init__(self):
        pass

    def exponentialDecay(
        self,
        initial_learning_rate=None,
        decay_steps=None,
        decay_rate=None,
        staircase=None,
        name=None,
    ):

        return kos.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase,
            name=name,
        )

    def piecewiseConstantDecay(self, boundaries=None, values=None, name=None):

        return kos.PiecewiseConstantDecay(
            boundaries=boundaries, values=values, name=name
        )

    def polynomialDecay(
        self,
        initial_learning_rate=None,
        decay_steps=None,
        end_learning_rate=None,
        power=None,
        cycle=None,
        name=None,
    ):

        return kos.PolynomialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            end_learning_rate=end_learning_rate,
            power=power,
            cycle=cycle,
            name=name,
        )

    def inverseTimeDecay(
        self,
        initial_learning_rate=None,
        decay_steps=None,
        decay_rate=None,
        staircase=None,
        name=None,
    ):

        return kos.InverseTimeDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase,
            name=name,
        )

    def cosineDecay(
        self,
        initial_learning_rate=None,
        decay_steps=None,
        alpha=None,
        name=None,
        warmup_target=None,
        warmup_steps=None,
    ):

        return kos.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            alpha=alpha,
            name=name,
            warmup_target=warmup_target,
            warmup_steps=warmup_steps,
        )

    def cosineDecayRestarts(
        self,
        initial_learning_rate=None,
        first_decay_steps=None,
        t_mul=None,
        m_mul=None,
        alpha=None,
        name=None,
    ):

        return kos.CosineDecayRestarts(
            initial_learning_rate=initial_learning_rate,
            first_decay_steps=first_decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha,
            name=name,
        )
