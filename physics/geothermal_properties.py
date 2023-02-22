import numpy as np
from darts.physics import *

#  Density dependent on compressibility only
class custom_density(property_evaluator_iface):
     def __init__(self, density, compressibility=0, p_ref=1):
        super().__init__()
        # Density evaluator class based on simple first order compressibility approximation (Taylor expansion)
        self.density_rc = density
        self.cr = compressibility
        self.p_ref = p_ref

     def evaluate(self, state):
        pres = state[0]
        return self.density_rc * (1 + self.cr * (pres - self.p_ref))

#  Viscosity dependent on viscosibility only
class custom_viscosity(property_evaluator_iface):
    def __init__(self, viscosity, viscosibility=0, p_ref=1):
        super().__init__()
        # Viscosity evaluator class based on simple first order approximation (Taylor expansion)
        self.viscosity_rc = viscosity
        self.dvis = viscosibility
        self.p_ref = p_ref

    def evaluate(self, state):
        pres = state[0]
        return self.viscosity_rc * (1 + self.dvis * (pres - self.p_ref))

#  Relative permeability based on Corey function
class custom_rel_perm(property_evaluator_iface):
    def __init__(self, exp, sr=0):
        super().__init__()
        self.exp = exp
        self.sr = sr

    def evaluate(self, sat):
        return (sat - self.sr)**self.exp

#  Flash based on constant K-values
class custom_flash(property_evaluator_iface):
    def __init__(self, K_values):
        super().__init__()
        # Custom flash class based on simple k-values Rachford-Rice solution
        self.K_values = np.array(K_values)

    def evaluate(self, state):
        pres = state[0]
        zc = np.array(state[1:])
        zc = np.append(zc, [1-np.sum(zc)])
        eps = 1e-12

        beta_min = 1 / (1 - np.max(self.K_values)) + eps
        beta_max = 1 / (1 - np.min(self.K_values)) - eps
        beta = 0.5 * (beta_min + beta_max)
        tol = 1e-12
        max_iter = 1000
        left = self.RR(beta_min, zc)

        for c in range(0, max_iter):
            if (np.abs(self.RR(beta, zc)) < tol):
                break
            if (left * self.RR(beta, zc) < 0):
                beta_max = beta
            else:
                beta_min = beta

            beta = 0.5 * (beta_min + beta_max)

        if c < max_iter:
            x = zc / (1 + beta * (self.K_values - 1))
            y = self.K_values * x
        else:
            x = zc
            print('Flash does not converged!\n')

        return beta, x, y

    def RR(self, beta, zc):
        return np.sum(zc * (1 - self.K_values) / (1 + beta * (self.K_values - 1)))

#  Chemistry for simplest kinetic rate
class custom_chemistry(property_evaluator_iface):
    def __init__(self, kin_rate, wat_molal, equi_prod, stoich_matrix, min_surf_area=1, min_z=1e-8, order_react=1):
        super().__init__()
        # Simple kinetic rate evaluator class
        self.kin_rate = kin_rate
        self.min_surf_area = min_surf_area
        self.order_react = order_react
        self.wat_molal = wat_molal
        self.equi_prod = equi_prod
        self.stoich_matrix = stoich_matrix
        self.z_min = min_z

    def evaluate(self, state):
        # Most simple kinetic reaction C1 --> H2O (note: this is unphysical obviously, but it's about the principle!)
        pres = state[0]

        vec_comp = np.array(state[1:])
        vec_comp = np.append(vec_comp, [1-np.sum(vec_comp)])

        # Store each kinetic operator:
        sol_prod = vec_comp[1]

        kinetic_rate = self.min_surf_area * self.kin_rate * (1 - (sol_prod/self.equi_prod)**self.order_react) * \
                       (vec_comp[1] - self.z_min)

        return kinetic_rate
