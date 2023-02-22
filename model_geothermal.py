from physics.geothermal import Geothermal
from darts.models.physics.iapws.iapws_property import *
from darts.models.physics.iapws.custom_rock_property import *
from model_base_rrm import RRMBaseModel
from darts.engines import value_vector
from physics.property_container import *


class Model(RRMBaseModel):

    def __init__(self,filename='',DEPTH=2000):
        # call base class constructor
        super().__init__(filename,DEPTH)

        self.hcap = np.array(self.reservoir.mesh.heat_capacity, copy=False)
        self.conduction = np.array(self.reservoir.mesh.rock_cond, copy=False)
        self.hcap.fill(2200)
        self.conduction.fill(181.44)

        # set initial pressure
        self.pressure_ini = DEPTH * 0.1 # bar

        # Create property containers:
        self.property_container = model_properties(phases_name=['water', 'steam', 'temperature', 'energy'],
                                                   components_name=['H2O'])


        # Define properties in property_container (IAPWS is the default property package for Geothermal in DARTS)
        # Users can define their custom properties in custom_properties.py; several property examples are defined there.
        self.rock = [value_vector([1, 0, 273.15])]
        self.property_container.temp_ev = iapws_temperature_evaluator()
        self.property_container.enthalpy_ev = dict([('water', iapws_water_enthalpy_evaluator()),
                                                    ('steam', iapws_steam_enthalpy_evaluator())])
        self.property_container.saturation_ev = dict([('water', iapws_water_saturation_evaluator()),
                                                    ('steam', iapws_steam_saturation_evaluator())])
        self.property_container.rel_perm_ev = dict([('water', iapws_water_relperm_evaluator()),
                                                    ('steam', iapws_steam_relperm_evaluator())])
        self.property_container.density_ev = dict([('water', iapws_water_density_evaluator()),
                                                   ('steam', iapws_steam_density_evaluator())])
        self.property_container.viscosity_ev = dict([('water', iapws_water_viscosity_evaluator()),
                                                     ('steam', iapws_steam_viscosity_evaluator())])
        self.property_container.saturation_ev = dict([('water', iapws_water_saturation_evaluator()),
                                                      ('steam', iapws_steam_saturation_evaluator())])

        self.property_container.rock_compaction_ev = custom_rock_compaction_evaluator(self.rock)
        self.property_container.rock_energy_ev = custom_rock_energy_evaluator(self.rock)

        self.reservoir.set_boundary_volume(xz_plus=1e16) #xz_minus=1e16, xz_plus=1e16, yz_minus=1e16,


        self.physics = Geothermal(property_container=self.property_container, timer=self.timer, n_points=64, min_p=100,
                                  max_p=400, min_e=1000, max_e=10000, grav=True)

        # Define time stepping criteria (first timestep 0.01 days, maximum timestep size 30 days) - keep default values
        self.params.first_ts = 1e-4
        self.params.mult_ts = 4
        self.params.max_ts = 365

        # Newton tolerance is relatively high because of L2-norm for residual and well segments (keep default values)
        self.params.tolerance_newton = 1e-3
        self.params.tolerance_linear = 1e-5
        self.params.max_i_newton = 20
        self.params.max_i_linear = 30
        self.runtime = 365*10
        self.inj = value_vector([0.999])

    def set_initial_conditions(self):
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, uniform_pressure=self.pressure_ini,
                                                    uniform_temperature=273.15+75)

    def set_boundary_conditions(self):
        for i, w in enumerate(self.reservoir.wells):
            if "INJ" in w.name:
                # w.control = self.physics.new_bhp_water_inj(360, 273.15+35)
                w.control = self.physics.new_rate_water_inj(9600, 273.15 + 30)
                w.constraint = self.physics.new_bhp_water_inj(min(self.reservoir.mesh.pressure)*1.3, 273.15 + 30)
            else:
                w.control = self.physics.new_rate_water_prod(9600)
                # w.control = self.physics.new_bhp_prod(240)

    def enthalpy_to_temperature(self, data):
        from darts.models.physics.iapws.iapws_property_vec import _Backward1_T_Ph_vec
        data_len = int(len(data) / 2)
        T = np.zeros(data_len)
        T[:] = _Backward1_T_Ph_vec(data[::2] / 10, data[1::2] / 18.015)
        return T

    def export_pro_vtk(self, file_name='temperature'):
        from darts.models.physics.iapws.iapws_property_vec import _Backward1_T_Ph_vec
        nb = self.reservoir.nb
        T = np.zeros(nb)
        X = np.array(self.physics.engine.X)
        T[:] = _Backward1_T_Ph_vec(X[:nb * 2:2] / 10, X[1:nb * 2:2] / 18.015)
        self.export_vtk(file_name=file_name, local_cell_data={'temp': T})

class model_properties(PropertyContainer):
    def __init__(self, phases_name, components_name):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        super().__init__(phases_name, components_name, Mw)

        # remove the virtual phase from the parent class
        self.dens = np.zeros(self.nph-2)
        self.sat = np.zeros(self.nph-2)
        self.mu = np.zeros(self.nph-2)
        self.kr = np.zeros(self.nph-2)
        self.enthalpy = np.zeros(self.nph-2)

    def evaluate(self, state):
        vec_state_as_np = np.asarray(state)

        for j in range(self.nph-2):
            self.enthalpy[j] = self.enthalpy_ev[self.phases_name[j]].evaluate(state)
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(state)
            self.sat[j] = self.saturation_ev[self.phases_name[j]].evaluate(state)
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(state)
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(state)

        self.temp = self.temp_ev.evaluate(state)
        self.rock_compaction = self.rock_compaction_ev.evaluate(state)
        self.rock_int_energy = self.rock_energy_ev.evaluate(state)

        return self.enthalpy, self.dens, self.sat, self.kr, self.mu, self.temp, self.rock_compaction, self.rock_int_energy
