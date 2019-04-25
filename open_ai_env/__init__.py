######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: April 17, 2019
#                      Author: Munir Jojo-Verge
#######################################################################

from gym.envs.registration import register
register(
    id='LG-SIM-ENV-v0',
    entry_point='open_ai_env.envs:LG_Sim_Env',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 40}
)

