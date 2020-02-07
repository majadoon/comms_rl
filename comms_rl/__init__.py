from gym.envs.registration import register

register(
    id='CommsRLTimeFreqResourceAllocation-v0',
    entry_point='comms_rl.envs:CommsRLTimeFreqResourceAllocationV0',
)
