from darts.engines import *
from darts.models.physics.iapws.iapws_property import *
from darts.models.physics.iapws.custom_rock_property import *
import numpy as np

class acc_flux_custom_iapws_evaluator_python(operator_set_evaluator_iface):
    def __init__(self, property_container):
        super().__init__()
        self.property = property_container

    def evaluate(self, state, values):
        pressure = state[0]

        (self.enthalpy, self.dens, self.sat, self.kr, self.mu, self.temp, self.rock_compaction,
         self.rock_int_energy) = self.property.evaluate(state)
        # mass accumulation
        values[0] = self.rock_compaction * np.sum(self.dens * self.sat)
        # mass flux
        values[1] = np.sum(self.dens * self.kr / self.mu)
        # fluid internal energy = water_enthalpy + steam_enthalpy - work
        # (in the following expression, 100 denotes the conversion factor from bars to kJ/m3)
        values[2] = self.rock_compaction * (np.sum(self.dens * self.sat * self.enthalpy) - 100 * pressure)
        # rock internal energy
        values[3] = self.rock_int_energy/self.rock_compaction
        # energy flux
        values[4] = np.sum(self.enthalpy * self.dens * self.kr / self.mu)
        # fluid conduction
        values[5] = 0 #181.44
        # rock conduction
        values[6] = 1 / self.rock_compaction
        # temperature
        values[7] = self.temp

        return 0


class acc_flux_custom_iapws_evaluator_python_well(operator_set_evaluator_iface):
    def __init__(self, property_container):
        super().__init__()
        self.property = property_container

    def evaluate(self, state, values):
        pressure = state[0]

        (self.enthalpy, self.dens, self.sat, self.kr, self.mu, self.temp, self.rock_compaction,
         self.rock_int_energy) = self.property.evaluate(state)
        # mass accumulation
        values[0] = self.rock_compaction * np.sum(self.dens * self.sat)
        # mass flux
        values[1] = np.sum(self.dens * self.kr / self.mu)
        # fluid internal energy = water_enthalpy + steam_enthalpy - work
        # (in the following expression, 100 denotes the conversion factor from bars to kJ/m3)
        values[2] = self.rock_compaction * (np.sum(self.dens * self.sat * self.enthalpy) - 100 * pressure)
        # rock internal energy
        values[3] = self.rock_int_energy / self.rock_compaction
        # energy flux
        values[4] = np.sum(self.enthalpy * self.dens * self.kr / self.mu)
        # fluid conduction
        values[5] = 181.44
        # rock conduction
        values[6] = 1 / self.rock_compaction
        # temperature
        values[7] = self.temp

        return 0


class acc_flux_gravity_evaluator_python(operator_set_evaluator_iface):
    def __init__(self, property_container, n_ops):
        super().__init__()
        self.property = property_container
        self.n_ops = n_ops

    def evaluate(self, state, values):
        pressure = state[0]

        for i in range(self.n_ops):
            values[i] = 0

        (self.enthalpy, self.dens, self.sat, self.kr, self.mu, self.temp, self.rock_compaction, self.rock_int_energy) = \
            self.property.evaluate(state)
        # mass accumulation
        ph = self.property.nph - 2
        values[0] = self.rock_compaction * np.sum(self.dens * self.sat)

        # mass flux
        shift = 1
        for i in range(ph):
            values[shift + i] = self.dens[i] * self.kr[i] / self.mu[i]

        shift = 3
        # fluid internal energy
        values[shift] = self.rock_compaction * (np.sum(self.dens * self.sat * self.enthalpy) - 100 * pressure)

        # rock internal energy
        shift += 1
        values[shift] = self.rock_int_energy/self.rock_compaction

        # energy flux
        shift += 1
        for i in range(ph):
            values[shift + i] = self.enthalpy[i] * self.dens[i] * self.kr[i] / self.mu[i]
        shift += 2
        # fluid conduction
        values[shift] = 2.0 * 86.4
        # rock conduction
        values[shift + 1] = 1 / self.rock_compaction
        # temperature
        values[shift + 2] = self.temp
        # density operator
        for i in range(ph):
            values[shift + 3 + i] = 0

        # print('State:', state, 'Ops:', values)

        return 0


class acc_flux_gravity_evaluator_python_well(operator_set_evaluator_iface):
    def __init__(self, property_container, n_ops):
        super().__init__()
        self.property = property_container
        self.n_ops = n_ops

    def evaluate(self, state, values):
        pressure = state[0]
        for i in range(self.n_ops):
            values[i] = 0

        (self.enthalpy, self.dens, self.sat, self.kr, self.mu, self.temp, self.rock_compaction, self.rock_int_energy) = \
            self.property.evaluate(state)
        # mass accumulation
        ph = self.property.nph - 2
        values[0] = self.rock_compaction * np.sum(self.dens * self.sat)

        # mass flux
        shift = 1
        for i in range(ph):
            values[shift + i] = self.dens[i] * self.kr[i] / self.mu[i]

        shift = 3
        # fluid internal energy
        values[shift] = self.rock_compaction * (np.sum(self.dens * self.sat * self.enthalpy) - 100 * pressure)

        # rock internal energy
        shift += 1
        values[shift] = self.rock_int_energy / self.rock_compaction

        # energy flux
        shift += 1
        for i in range(ph):
            values[shift + i] = self.enthalpy[i] * self.dens[i] * self.kr[i] / self.mu[i]
        shift += 2
        # fluid conduction
        values[shift] = 0.0 * 86.4
        # rock conduction
        values[shift + 1] = 1 / self.rock_compaction
        # temperature
        values[shift + 2] = self.temp
        # density operator
        for i in range(ph):
            values[shift + 3 + i] = 0

        # print('State:', state, 'Ops:', values)

        return 0

class geothermal_rate_custom_evaluator_python(operator_set_evaluator_iface):
    def __init__(self, property_container):
        super().__init__()
        self.property = property_container

    def evaluate(self, state, values):
        pressure = state[0]

        (self.enthalpy, self.dens, self.sat, self.kr, self.mu, self.temp, self.rock_compaction, self.rock_int_energy) = \
            self.property.evaluate(state)
        # mass accumulation
        ph = self.property.nph - 2

        total_density = np.sum(self.sat * self.dens)
        total_flux = np.sum(self.dens * self.kr / self.mu)

        for i in range(ph):
            values[i] = self.sat[i] * total_flux / total_density

        # temperature
        values[2] = self.temp
        # energy rate
        values[3] = np.sum(self.enthalpy * self.dens * self.kr / self.mu)

        return 0
        
class geothermal_mass_rate_custom_evaluator_python(operator_set_evaluator_iface):
    def __init__(self, property_data):
        super().__init__()

        self.water_density      = property_data.water_density
        self.steam_density      = property_data.steam_density
        self.water_saturation   = property_data.water_saturation
        self.steam_saturation   = property_data.steam_saturation
        self.water_relperm      = property_data.water_relperm
        self.steam_relperm      = property_data.steam_relperm		
        self.water_viscosity    = property_data.water_viscosity
        self.steam_viscosity    = property_data.steam_viscosity
        self.temperature        = property_data.temperature
        self.water_enth         = property_data.water_enthalpy
        self.steam_enth         = property_data.steam_enthalpy

		
    def evaluate(self, state, values):
        water_den = self.water_density.evaluate(state)
        steam_den = self.steam_density.evaluate(state)
        water_sat = self.water_saturation.evaluate(state)
        steam_sat = self.steam_saturation.evaluate(state)
        water_rp  = self.water_relperm.evaluate(state)
        steam_rp  = self.steam_relperm.evaluate(state)
        water_vis = self.water_viscosity.evaluate(state)
        steam_vis = self.steam_viscosity.evaluate(state)
        temp      = self.temperature.evaluate(state)
        water_enth= self.water_enth.evaluate(state)
        steam_enth= self.steam_enth.evaluate(state)

        total_density = water_sat * water_den + steam_sat * steam_den

        # water mass rate
        values[0] = water_den * water_rp / water_vis + steam_den * steam_rp / steam_vis
        # steam mass rate
        values[1] = steam_sat * (water_den * water_rp / water_vis + steam_den * steam_rp / steam_vis) / total_density
        # temperature
        values[2] = temp
        # energy rate
        values[3] = water_enth * water_den * water_rp / water_vis + steam_enth * steam_den * steam_rp / steam_vis
        
        return 0