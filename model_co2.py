from model_base_rrm import RRMBaseModel
from darts.engines import sim_params, value_vector
from physics.physics_comp_sup import SuperPhysics
import numpy as np
from physics.properties_basic import *
from physics.property_container import *
from physics.operator_evaluator_sup import DefaultPropertyEvaluator

# Model class creation here!
class Model(RRMBaseModel):

    def __init__(self,filename='',DEPTH=2000, temperature= None):
        # call base class constructor
        super().__init__(filename,DEPTH)

        self.zero = 1e-8
        """Physical properties"""
        # Create property containers:
        # components_name = ['CO2', 'C1', 'H2O']
        self.components = ["CO2",  "H2O"]
        self.phases = ["gas", "wat"]
        Mw = [44.01,  18.015]
        # for name in components_name:
        #     Mw.append(props(name, 'Mw'))
        self.property_container = PropertyContainer(phases_name=self.phases,
                                                     components_name=self.components,
                                                     Mw=Mw, min_z=self.zero / 10, temperature=temperature)

        self.nc = self.ne = len(self.components)
        self.vars = ['P'] + self.components[:-1]
        self.thermal = False
        if self.thermal:
            self.vars += ['T']
            self.ne += 1

        """ properties correlations """
        # Simple properties
        self.property_container.flash_ev = Flash(self.components, [4, 1e-1], self.zero)
        self.property_container.density_ev = dict([('gas', Density(self.components, compr=1e-3, dens0=200)),
                                                   ('wat',
                                                    Density(self.components, compr=1e-5, dens0=1000, x_mult=320))])
        self.property_container.viscosity_ev = dict([('gas', ViscosityConst(0.015)),
                                                     ('wat', ViscosityConst(1.0))])
        self.property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas")),
                                                    ('wat', PhaseRelPerm("oil"))])
        self.property_container.enthalpy_ev = dict([('gas', Enthalpy(hcap=2.2e-3 * 20)),
                                                    ('wat', Enthalpy(hcap=4.2e-3 * 20))])
        self.property_container.conductivity_ev = dict([('gas', ConstProp(0.034 * 86.4)),
                                                        ('wat', ConstProp(0.6 * 86.4))])

        """ Activate physics """
        self.physics = SuperPhysics(self.property_container, self.timer, n_points=200, min_p=1, max_p=400,
                                    min_z=self.zero/10, max_z=1-self.zero/10)

        # min_P, max_P = 1., 400.
        # min_T, max_T = 273.15, 423.15
        # min_z, max_z = self.zero/10, 1-self.zero/10
        # self.output_props = PropertyEvaluator(self.vars, self.property_container)
        # self.physics = SuperPhysics(self.property_container, self.timer, n_points=500,
        #                             min_p=min_P, max_p=max_P, min_t=min_T, max_t=max_T, min_z=min_z, max_z=max_z,
        #                             out_props=self.output_props, thermal=self.thermal, cache=True)

        self.inj_comp = [1.0 - 2 * self.zero]
        self.ini_comp = [0.005]

        self.pressure_ini = DEPTH * 0.1 # bar
        self.temp_ini = temperature

        # Some newton parameters for non-linear solution:
        self.params.first_ts = 1e-5
        self.params.mult_ts = 1.5
        self.params.max_ts = 5

        self.params.tolerance_newton = 1e-3
        self.params.tolerance_linear = 1e-4
        self.params.max_i_newton = 10
        self.params.max_i_linear = 50
        self.params.newton_type = sim_params.newton_local_chop

        self.reservoir.set_boundary_volume(xz_plus=1e15)

    # Initialize reservoir and set boundary conditions:
    def set_initial_conditions(self):
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, self.pressure_ini, self.ini_comp)

    def set_op_list(self):
        self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
        n_res = self.reservoir.mesh.n_res_blocks
        self.op_num[n_res:] = 1
        self.op_list = [self.physics.acc_flux_itor, self.physics.acc_flux_w_itor]

    def export_pro_vtk(self, file_name='Saturation'):
        Xn = np.array(self.physics.engine.X, copy=False)
        P = Xn[0:self.reservoir.nb * 2:2]
        z1 = Xn[1:self.reservoir.nb * 2:2]
        # z2 = Xn[2:self.reservoir.nb * 3:3]

        sg = np.zeros(len(P))
        sw = np.zeros(len(P))

        for i in range(len(P)):
            values = value_vector([0] * self.physics.n_ops)
            # state = value_vector((P[i], z1[i], z2[i]))
            state = value_vector((P[i], z1[i]))
            self.physics.property_itor.evaluate(state, values)
            sg[i] = values[0]
            sw[i] = 1 - sg[i]

        self.export_vtk(file_name, local_cell_data={'GasSat': sg, 'WatSat': sw})

    def set_boundary_conditions(self):
        for i, w in enumerate(self.reservoir.wells):
            if "INJ" in w.name:
                w.control = self.physics.new_bhp_inj(self.pressure_ini*1.25, self.inj_comp)
                w.constraint = self.physics.new_bhp_inj(min(self.reservoir.mesh.pressure) * 1.3, self.inj_comp)

            else:
                w.control = self.physics.new_rate_prod(0, 1)
                #w.control = self.physics.new_bhp_prod(300)




class PropertyEvaluator(DefaultPropertyEvaluator):
    def __init__(self, variables, property_container):
        super().__init__(variables, property_container)  # Initialize base-class

        self.props = ['sat_g', 'sat_w']
        self.n_props = len(self.props)

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        (self.sat, self.x, rho, self.rho_m, self.mu, rates, self.kr, self.pc, self.ph) = self.property.evaluate(state)

        values[0] = self.sat[0]
        values[1] = self.sat[1]

        return 0
