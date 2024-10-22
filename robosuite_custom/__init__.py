# flake8: noqa


from robosuite.environments.base import make, register_env

from robosuite_custom.environments.peg_in_hole import PegInHoleEnv
from robosuite_custom.environments.plane_peg_in_hole import PlanePegInHoleEnv
from robosuite_custom.wrappers.data_collection_wrapper import (
    DataCollectionWrapper,
)
from robosuite_custom.wrappers.gym_wrapper import GymWrapper

# from robosuite_custom.wrappers.encoder_warpper import EncoderWrapper

# Register so that robosuite can recognize them
register_env(PegInHoleEnv)
register_env(PlanePegInHoleEnv)
