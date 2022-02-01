import numpy as np


def seasyhrva_vsy_vh_vd_odes(t, x, b_a, b_sy, g_a, g_sy, g_h, e_a, e_sy, a, h, v, b_a_v, b_sy_v, g_a_v, g_sy_v, g_h_v, a_v, h_v, d_h, d_h_v):    
    S = x[0]  # Susceptible
    E = x[1]  # Exposed
    A = x[2]  # Infected Asymptomatic
    Sy = x[3]  # Infected Symptomatic
    A_v = x[4]  # Infected Asymptomatic Vaccinated
    Sy_v = x[5]  # Infected Symptomatic Vaccinated
    H = x[6]  # Hospitalised
    H_v = x[7]  # Hospitalised Vaccinated
    R = x[8]  # Recovered
    V = x[9]  # Vaccinated
    D = x[10]  # Deceased
    
    N = np.sum([S, E, A, Sy, A_v, Sy_v, H, H_v, R, V])  # Dynamic population (excludes deceased people)
    
    dSdt = -(b_a/N)*(A+A_v)*S -(b_sy/N)*(Sy+Sy_v)*S - v*S
    dEdt = (b_a/N)*(A+A_v)*S + (b_sy/N)*(Sy+Sy_v)*S - e_a*E - e_sy*E
    dAdt = e_a*E - g_a*A - a*A
    dSydt = e_sy*E + a*A - g_sy*Sy - h*Sy
    dA_vdt = (b_a/N)*(A+A_v)*V - a_v*A_v - g_a_v*A_v
    dSy_vdt = (b_sy/N)*(Sy+Sy_v)*V + a_v*A_v - h_v*Sy_v - g_sy_v*Sy_v
    dHdt = h*Sy - g_h*H - d_h*H
    dH_vdt = h_v*Sy_v - g_h_v*H_v - d_h_v*H_v
    dRdt = g_a*A + g_sy*Sy + g_h*H + g_h_v*H_v + g_a_v*A_v + g_sy_v*Sy_v
    dVdt = v*S -(b_a/N)*(A+A_v)*V -(b_sy/N)*(Sy+Sy_v)*V
    dDdt = d_h*H + d_h_v*H_v

    return [dSdt, dEdt, dAdt, dSydt, dA_vdt, dSy_vdt, dHdt, dH_vdt, dRdt, dVdt, dDdt]


def seasyhrva_vsy_vh_vd_odes_noise(t, x, b_a, b_sy, g_a, g_sy, g_h, e_a, e_sy, a, h, v, b_a_v, b_sy_v, g_a_v, g_sy_v, g_h_v, a_v, h_v, d_h, d_h_v, std_f=0):    
    S = x[0]  # Susceptible
    E = x[1]  # Exposed
    A = x[2]  # Infected Asymptomatic
    Sy = x[3]  # Infected Symptomatic
    A_v = x[4]  # Infected Asymptomatic Vaccinated
    Sy_v = x[5]  # Infected Symptomatic Vaccinated
    H = x[6]  # Hospitalised
    H_v = x[7]  # Hospitalised Vaccinated
    R = x[8]  # Recovered
    V = x[9]  # Vaccinated
    D = x[10]  # Deceased

    N = np.sum([S, E, A, Sy, A_v, Sy_v, H, H_v, R, V])  # Dynamic population (excludes deceased people)
    
    b_a = abs(np.random.normal(b_a, std_f*b_a))
    b_sy = abs(np.random.normal(b_sy, std_f*b_sy))
    g_a = abs(np.random.normal(g_a, std_f*g_a))
    g_sy = abs(np.random.normal(g_sy, std_f*g_sy))
    g_h = abs(np.random.normal(g_h, std_f*g_h))
    e_a = abs(np.random.normal(e_a, std_f*e_a))
    e_sy = abs(np.random.normal(e_sy, std_f*e_sy))
    a = abs(np.random.normal(a, std_f*a))
    h = abs(np.random.normal(h, std_f*h))
    v = abs(np.random.normal(v, std_f*v))
    b_a_v = abs(np.random.normal(b_a_v, std_f*b_a_v))
    b_sy_v = abs(np.random.normal(b_sy_v, std_f*b_sy_v))
    g_a_v = abs(np.random.normal(g_a_v, std_f*g_a_v))
    g_sy_v = abs(np.random.normal(g_sy_v, std_f*g_sy_v))
    g_h_v = abs(np.random.normal(g_h_v, std_f*g_h_v))
    a_v = abs(np.random.normal(a_v, std_f*a_v))
    h_v = abs(np.random.normal(h_v, std_f*h_v))
    d_h = abs(np.random.normal(d_h, std_f*d_h))
    d_h_v = abs(np.random.normal(d_h_v, std_f*d_h_v))

    dSdt = -(b_a/N)*(A+A_v)*S -(b_sy/N)*(Sy+Sy_v)*S - v*S
    dEdt = (b_a/N)*(A+A_v)*S + (b_sy/N)*(Sy+Sy_v)*S - e_a*E - e_sy*E
    dAdt = e_a*E - g_a*A - a*A
    dSydt = e_sy*E + a*A - g_sy*Sy - h*Sy
    dA_vdt = (b_a/N)*(A+A_v)*V - a_v*A_v - g_a_v*A_v
    dSy_vdt = (b_sy/N)*(Sy+Sy_v)*V + a_v*A_v - h_v*Sy_v - g_sy_v*Sy_v
    dHdt = h*Sy - g_h*H - d_h*H
    dH_vdt = h_v*Sy_v - g_h_v*H_v - d_h_v*H_v
    dRdt = g_a*A + g_sy*Sy + g_h*H + g_h_v*H_v + g_a_v*A_v + g_sy_v*Sy_v
    dVdt = v*S -(b_a/N)*(A+A_v)*V -(b_sy/N)*(Sy+Sy_v)*V
    dDdt = d_h*H + d_h_v*H_v

    return [dSdt, dEdt, dAdt, dSydt, dA_vdt, dSy_vdt, dHdt, dH_vdt, dRdt, dVdt, dDdt]


models = {
    'seasyhrva_vsy_vh_vd':seasyhrva_vsy_vh_vd_odes,
    'seasyhrva_vsy_vh_vd_noise':seasyhrva_vsy_vh_vd_odes_noise
}
