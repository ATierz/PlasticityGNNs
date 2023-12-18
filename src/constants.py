import numpy as np

NUM_SIMULATIONS = 10
Rs = np.round(np.linspace(1, 1., NUM_SIMULATIONS), decimals=2).tolist()  # [1, 1.25, 1.5, 1.75, 2, 2.5]
Ls = np.round(np.linspace(4, 1, NUM_SIMULATIONS), decimals=2).tolist()  # [4, 3.5, 3, 2.75, 2, 1.75]

STATE_VARIABLES = ['Node Label','U.Magnitude', 'U.U1', 'U.U2', 'U.U3','V.Magnitude', 'V.V1', 'V.V2', 'V.V3', 'E' ,'S.Mises']

test_index = np.random.choice(np.arange(0, NUM_SIMULATIONS, 1), NUM_SIMULATIONS//4)
TRAIN_SIMULATIONS = [f'rectangle_L{Ls[i]}_R{Rs[i]}' for i in range(NUM_SIMULATIONS) if i not in test_index]
TEST_SIMULATIONS = [f'rectangle_L{Ls[i]}_R{Rs[i]}' for i in test_index]