from .environment import EnvironmentStepTypes
from .reinforcement.frozen_lake_environment import (make_frozen_lake_4x4_env,
                                                    make_frozen_lake_8x8_env)
from .supervised.classification_environment import ClassificationEnvironment
from .supervised.multiplexer.multiplexer_factories import (
    make_discrete_mux_env, make_real_mux_env)
