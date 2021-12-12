from run_model import run_model
import numpy as np


def generate_data(model_params, model_state_params):
    generated_model = np.array(run_model('sirv_c_noise', model_params, model_state_params)[0])
    S, E, A, Sy, H, R, D, V, A_v, Sy_v, H_v = generated_model
    N = S + E + A + Sy + H + R + V + A_v + Sy_v + H_v

    return generated_model, N