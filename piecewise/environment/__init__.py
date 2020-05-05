from .environment import EnvironmentStepTypes
from .reinforcement.cartpole_environment import make_cartpole_env
from .reinforcement.mountain_car_environment import make_mountain_car_env
from .reinforcement.frozen_lake_environment import (
    make_frozen_lake_8x8_test_env, make_frozen_lake_8x8_train_env,
    make_frozen_lake_12x12_test_env, make_frozen_lake_12x12_train_env)
