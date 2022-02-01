from run_model import run_model
import numpy as np


def generate_data_seasyhrva_vsy_vh_vd(model_params, model_state_params):
    generated_model = np.array(run_model('seasyhrva_vsy_vh_vd_noise', model_params, model_state_params)[0])
    S, E, A, Sy, A_v, Sy_v, H, H_v, R, V, D = generated_model
    N = S + E + A + Sy + A_v + Sy_v + H + H_v + R + V

    return generated_model, N
