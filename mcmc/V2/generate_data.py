from run_model import run_model
import numpy as np


def generate_data_seirv(model_params, model_state_params):
    generated_model = np.array(run_model('seirv_noise', model_params, model_state_params)[0])
    S, E, I, R, V = generated_model
    N = S + E + I + R + V

    return generated_model, N
