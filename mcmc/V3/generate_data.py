from run_model import run_model
import numpy as np


def generate_data_seirvd(model_params, model_state_params):
    generated_model = np.array(run_model('seirvd_noise', model_params, model_state_params)[0])
    S, E, I, R, V, D = generated_model
    N = S + E + I + R + V

    return generated_model, N
