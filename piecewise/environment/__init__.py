from .environment import EnvironmentStepTypes
from .reinforcement.cartpole_environment import make_continuous_cartpole_env
from .reinforcement.mountain_car_environment import \
    make_continuous_mountain_car_env
from .supervised.classification_environment import ClassificationEnvironment
from .supervised.multiplexer.multiplexer_factories import (
    make_discrete_mux_env, make_real_mux_env)
