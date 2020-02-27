from .environment import EnvironmentStepTypes
from .reinforcement.cartpole_environment import CartpoleEnvironment
from .reinforcement.gym_environment import (GymEnvironment,
                                            discretise_gym_environment)
from .supervised.classification_environment import ClassificationEnvironment
from .supervised.multiplexer.multiplexer_factories import (
    make_discrete_mux_env, make_real_mux_env)
