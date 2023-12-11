""" Written by Brian Hou for CSE571: Probabilistic Robotics (Winter 2019)
"""

import numpy as np

from utils import minimized_angle


class ExtendedKalmanFilter:
    def __init__(self, mean, cov, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.reset()

    def reset(self):
        self.mu = self._init_mean
        self.sigma = self._init_cov

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark
        observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        # YOUR IMPLEMENTATION HERE

        # PASO 1: Obtengo el valor de Q_t
        Q_t = env.beta

        # PASO 2, 3 y 4: se calculan los valores necesarios para las siguientes matrices
        alfa = self.alphas
        x = self.mu
        G_t = env.G(x,u)
        V_t = env.V(x,u)
        mu_techo = env.forward(x, u)
        M_t = env.noise_from_motion(u, alfa)
        R_t = V_t@M_t@V_t.T
        sigma_techo = G_t@self.sigma@G_t.T + R_t

        # PASO 5: se calcula la prediccion de la medida
        z_pred = env.observe(mu_techo,marker_id)

        # PASO 6: se halla la matriz H_t (Jacobiano del modelo de medida)
        H_t = env.H(x,marker_id)

        # PASO 7: se halla la matriz S
        S_t = H_t@sigma_techo@H_t.T + Q_t

        # PASO 8: se halla K (la ganacia del filtro de Kalman)
        K_t = sigma_techo@H_t.T@np.linalg.inv(S_t)

        # PASO 9: se calcula el nuevo mu_techo
        mu_techo = mu_techo + K_t@minimized_angle(z - z_pred)

        # PASO 10: se calcula el nuevo sigma_techo
        sigma_techo = (np.identity(K_t.shape[0]) - K_t@H_t)@sigma_techo
        
        # PASO 11: se actualizan mu y sigma
        self.mu = mu_techo
        self.sigma = sigma_techo

        return self.mu, self.sigma
