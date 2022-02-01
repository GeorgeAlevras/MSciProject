from run_model import run_model
import numpy as np


def generate_data_seasyhrvd(model_params, model_state_params):
    generated_model = np.array(run_model('seasyhrvd_noise', model_params, model_state_params)[0])
    S, E, A, Sy, H, R, V, D = generated_model
    N = S + E + A + Sy + H + R + V

    return generated_model, N
