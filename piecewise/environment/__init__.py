from .reinforcement.gym_environment import GymEnvironment
from .reinforcement.cartpole_environment import make_cartpole_environment
from .supervised.classification_environment import ClassificationEnvironment
from .supervised.multiplexer.multiplexer_factories import (
    make_discrete_mux_env, make_real_mux_env)
from .environment import EnvironmentStepTypes
