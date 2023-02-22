from physics.physics_comp_sup_do import SuperPhysics
from darts.engines import value_vector
from model_base_rrm import RRMBaseModel
import numpy as np
# from physics.property_container import *
from physics.properties_dead_oil import *
from physics.property_container_do import *


class Model(RRMBaseModel):

    def __init__(self,filename='',DEPTH=2000):
        # Call base class constructor
        super().__init__(filename, DEPTH)

        """Physical properties"""
        self.pvt = 'physics_do.in'
        self.zero = 1e-13
        self.property_container = model_properties(phases_name=['water', 'oil'],
                                                   components_name=['w', 'o'],
                                                   pvt=self.pvt, min_z=self.zero/10)

        # set initial pressure
        self.pressure_ini = DEPTH * 0.1 # bar

        # Define property evaluators based on custom properties
        self.flash_ev = []
        self.property_container.density_ev = dict([('water', DensityWat(self.pvt)),
                                                   ('oil', DensityOil(self.pvt))])
        self.property_container.viscosity_ev = dict([('water', ViscoWat(self.pvt)),
                                                     ('oil', ViscoOil(self.pvt))])
        self.property_container.rel_perm_ev = dict([('water', WatRelPerm(self.pvt)),
                                                    ('oil', OilRelPerm(self.pvt))])
        self.property_container.capillary_pressure_ev = CapillarypressurePcow(self.pvt)

        self.property_container.rock_compress_ev = RockCompactionEvaluator(self.pvt)

        # Set up physics for dead oil model
        self.thermal = 0
        self.physics = SuperPhysics(self.property_container, self.timer, n_points=400, min_p=20, max_p=1000,
                                     min_z=self.zero, max_z=1 - self.zero, thermal=self.thermal)

        # Settings for time steps and solver accuracy
        self.params.first_ts = 0.01
        self.params.mult_ts = 2
        self.params.max_ts = 100
        self.params.tolerance_newton = 1e-3
        self.params.tolerance_linear = 1e-4

        self.runtime = 300
        self.inj = value_vector([0.999])

        self.timer.node["initialization"].stop()

        # set large reservoir volume at the minimum and maximum of the x direction
        self.reservoir.set_boundary_volume(xz_minus=1e15)
        self.reservoir.set_boundary_volume(xz_plus=1e15)

    # set the initial reservoir pressure (in bars) and initial water saturation
    def set_initial_conditions(self):
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, uniform_pressure= self.pressure_ini,
                                                      uniform_composition=[0.2])

    # set the well controls for injector and producer
    def set_boundary_conditions(self):
        for i, w in enumerate(self.reservoir.wells):
            if "INJ" in w.name:
                w.control = self.physics.new_rate_inj(6000, self.inj, 0)
                w.constraint = self.physics.new_bhp_inj(min(self.reservoir.mesh.pressure)*1.3, self.inj)
            else:
                w.control = self.physics.new_rate_prod(6000, 1)
                w.constraint = self.physics.new_bhp_prod(150)

    # utility function to save model to VTK files for visualisation
    def export_pro_vtk(self, file_name='Saturation'):
        Xn = np.array(self.physics.engine.X, copy=False)
        P = Xn[0:self.reservoir.nb * 2:2]
        z1 = Xn[1:self.reservoir.nb * 2:2]

        so = np.zeros(len(P))
        sw = np.zeros(len(P))

        for i in range(len(P)):
            values = value_vector([0] * self.physics.n_ops)
            state = value_vector((P[i], z1[i]))
            self.physics.property_itor.evaluate(state, values)
            sw[i] = values[0]
            so[i] = 1 - sw[i]

        self.export_vtk(file_name, local_cell_data={'OilSat': so, 'WatSat': sw})

class model_properties(property_container):
    def __init__(self, phases_name, components_name, pvt, min_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        super().__init__(phases_name, components_name, Mw, min_z)
        self.x = np.zeros((self.nph, self.nc))
        self.pvt = pvt
        self.surf_dens = get_table_keyword(self.pvt, 'DENSITY')[0]
        self.surf_oil_dens = self.surf_dens[0]
        self.surf_wat_dens = self.surf_dens[1]

    def evaluate(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        zc = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))

        self.clean_arrays()
        # two-phase flash - assume water phase is always present and water component last
        for i in range(self.nph):
            self.x[i, i] = 1

        ph = [0, 1]

        for j in ph:
            M = 0
            # molar weight of mixture
            for i in range(self.nc):
                M += self.Mw[i] * self.x[j][i]
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(state)  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(state)  # output in [cp]

        self.nu = zc
        self.compute_saturation(ph)

        for j in ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[0])
            self.pc[j] = 0

        return self.sat, self.x, self.dens, self.dens_m, self.mu, self.kr, self.pc, ph

    def evaluate_at_cond(self, pressure, zc):

        self.sat[:] = 0

        state = value_vector([1, 0])

        ph = [0, 1]
        for j in ph:
            self.dens_m[j] = self.density_ev[self.phases_name[j]].evaluate(state)

        self.dens_m = [self.surf_wat_dens, self.surf_oil_dens]  # to match DO based on PVT

        self.nu = zc
        self.compute_saturation(ph)

        return self.sat, self.dens_m
