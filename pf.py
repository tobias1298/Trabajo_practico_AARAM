""" Written by Brian Hou for CSE571: Probabilistic Robotics (Winter 2019)
"""

import numpy as np

from utils import minimized_angle


class ParticleFilter:
    def __init__(self, mean, cov, num_particles, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.num_particles = num_particles
        self.reset()

    def reset(self):
        self.particles = np.zeros((self.num_particles, 3))
        for i in range(self.num_particles):
            self.particles[i, :] = np.random.multivariate_normal(
                self._init_mean.ravel(), self._init_cov)
        self.weights = np.ones(self.num_particles) / self.num_particles

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark
        observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        # YOUR IMPLEMENTATION HERE
        mean, cov = self.mean_and_variance(self.particles)
        
        M = self.num_particles
        X_techo = self.particles

        peso_techo = self.weights
        
        for m in range(M):
            # Estimo el movimiento con ruido dado el control u
            u_ruido = env.sample_noisy_action(u,self.alphas)
            
            # Se predice el nuevo estado de la particula usando el estado de control
            xtm = env.forward(X_techo[m,:],u_ruido)
            
            # se calcula la observacion que se tendria
            z_obs = env.sample_noisy_observation(xtm,marker_id,self.beta)# env.observe(xtm,marker_id)#

            # se calcula la probabilidad de esa medida
            peso_techo[m] = env.likelihood(minimized_angle(z_obs - z),self.beta)#

            # actualizo las particulas
            X_techo[m,:] = xtm.T #env.forward(X_techo[m,:],u_ruido).T
 
        tot = np.sum(peso_techo)
        peso_techo = peso_techo/tot

        # se decide si se aplica el paso de remuestreo
        remuestrar = np.all(X_techo != self.particles)#False #np.all(X_techo == self.particles)
        if remuestrar:
            mean, cov = self.mean_and_variance(X_techo)
        else:
            new_particles,_ = self.resample(X_techo,peso_techo)
            mean, cov = self.mean_and_variance(new_particles)
        
        return mean, cov

    def resample(self, particles, weights):
        """Sample new particles and weights given current particles and weights. Be sure
        to use the low-variance sampler from class.

        particles: (n x 3) matrix of poses
        weights: (n,) array of weights
        """
        new_particles, new_weights = particles, weights
        # YOUR IMPLEMENTATION HERE

        # PASO 2: definir el vector que va a tener las nuevas particulas
        # ya se hizo mas arriba

        # PASO 3: se selecciona el numero aleatorio que se va a utilizar en este algoritmo
        n = particles.shape[0]
    
        c = weights[0]
        
        r = np.random.rand()*(1/n)

        i = 1

        #tot = np.sum(new_weights)
        #new_weights = new_weights/tot
        for j in range(n):
            u = r + (j)*(1/n)
            while u > c and i < n-1: #
                i += 1
                c += weights[i]
            new_particles[j,:] = particles[i,:]
            #new_weights[j] = weights[i]#1/n

        return new_particles, new_weights

    def mean_and_variance(self, particles):
        """Compute the mean and covariance matrix for a set of equally-weighted
        particles.

        particles: (n x 3) matrix of poses
        """
        mean = particles.mean(axis=0)
        mean[2] = np.arctan2(
            np.cos(particles[:, 2]).sum(),
            np.sin(particles[:, 2]).sum()
        )

        zero_mean = particles - mean
        for i in range(zero_mean.shape[0]):
            zero_mean[i, 2] = minimized_angle(zero_mean[i, 2])
        cov = np.dot(zero_mean.T, zero_mean) / self.num_particles

        return mean.reshape((-1, 1)), cov
