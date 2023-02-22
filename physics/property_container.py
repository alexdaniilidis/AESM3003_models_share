import numpy as np


class PropertyContainer:
    def __init__(self, phases_name, components_name, Mw, min_z=1e-11,
                 diff_coef=0., rock_comp=1e-6, temperature=None):
        super().__init__()
        # This class contains all the property evaluators required for simulation
        if temperature is not None:  # constant T specified
            self.thermal = False
        else:
            self.thermal = True
        self.temperature = temperature

        self.nph = len(phases_name)
        self.nc = len(components_name)
        self.components_name = components_name
        self.phases_name = phases_name
        self.min_z = min_z
        self.Mw = Mw

        self.rock_comp = rock_comp
        self.p_ref = 1.0
        self.diff_coef = diff_coef

        # Allocate (empty) evaluators for functions
        self.density_ev = []
        self.viscosity_ev = []
        self.rel_perm_ev = []
        self.rel_well_perm_ev = []
        self.enthalpy_ev = []
        self.conductivity_ev = []
        self.rock_energy_ev = []
        self.capillary_pressure_ev = []
        self.kinetic_rate_ev = []
        self.heat_source_ev = None
        self.flash_ev = 0

        # passing arguments
        self.x = np.zeros((self.nph, self.nc))
        self.dens = np.zeros(self.nph)
        self.dens_m = np.zeros(self.nph)
        self.sat = np.zeros(self.nph)
        self.nu = np.zeros(self.nph)
        self.mu = np.zeros(self.nph)
        self.kr = np.zeros(self.nph)
        self.pc = np.zeros(self.nph)
        self.enthalpy = np.zeros(self.nph)
        self.cond = np.zeros(self.nph)

        self.phase_props = [self.dens, self.dens_m, self.sat, self.nu, self.mu, self.kr, self.pc, self.enthalpy, self.cond]

    def comp_out_of_bounds(self, vec_composition):
        # Check if composition sum is above 1 or element comp below 0, i.e. if point is unphysical:
        temp_sum = 0
        count_corr = 0
        check_vec = np.zeros((len(vec_composition),))

        for ith_comp in range(len(vec_composition)):
            if vec_composition[ith_comp] < self.min_z:
                #print(vec_composition)
                vec_composition[ith_comp] = self.min_z
                count_corr += 1
                check_vec[ith_comp] = 1
            elif vec_composition[ith_comp] > 1 - self.min_z:
                #print(vec_composition)
                vec_composition[ith_comp] = 1 - self.min_z
                temp_sum += vec_composition[ith_comp]
            else:
                temp_sum += vec_composition[ith_comp]

        for ith_comp in range(len(vec_composition)):
            if check_vec[ith_comp] != 1:
                vec_composition[ith_comp] = vec_composition[ith_comp] / temp_sum * (1 - count_corr * self.min_z)
        return vec_composition

    def clean_arrays(self):
        for a in self.phase_props:
            a[:] = 0
        for j in range(self.nph):
            self.x[j][:] = 0

    def run_flash(self, pressure, temperature, zc):
        self.nu, self.x = self.flash_ev.evaluate(pressure, temperature, zc)

        ph = []
        for j in range(self.nph):
            if self.nu[j] > 0:
                ph.append(j)

        if len(ph) == 1:
            self.x[ph[0]] = zc

        return ph

    def compute_saturation(self, ph):
        # Get saturations [volume fraction]
        Vtot = 0
        for j in ph:
            Vtot += self.nu[j] / self.dens_m[j]

        for j in ph:
            self.sat[j] = (self.nu[j] / self.dens_m[j]) / Vtot

        return

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

        zc = np.append(vec_state_as_np[1:self.nc], 1 - np.sum(vec_state_as_np[1:self.nc]))
        if zc[-1] < 0:
            # print(zc)
            zc = self.comp_out_of_bounds(zc)

        if self.thermal:
            temperature = vec_state_as_np[-1]
        else:
            temperature = self.temperature

        self.clean_arrays()

        # Flash
        self.ph = self.run_flash(pressure, temperature, zc)

        # Density and viscosity
        for j in self.ph:
            M = 0
            # molar weight of mixture (note: self.nc is fluid components here!)
            for i in range(self.nc):
                M += self.Mw[i] * self.x[j][i]
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, temperature, self.x[j, :])  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M  # molar density [kg/m3]/[kg/kmol]=[kmol/m3]
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(pressure, temperature, self.x[j, :], self.dens[j])  # output in [cp]

        # Saturations and relperms
        self.compute_saturation(self.ph)
        for j in self.ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[j])

        # Capillary pressure
        self.pc = np.zeros(self.nph)
        if self.capillary_pressure_ev:
            pc = self.capillary_pressure_ev.evaluate(self.sat[1])
            pc = np.append(pc, 0.)
            for j in self.ph:
                self.pc[j] = pc[j]

        # Kinetic rates
        rates = np.zeros(self.nc)
        for j, reaction in enumerate(self.kinetic_rate_ev):
            rate = reaction.evaluate(pressure, temperature, self.x, self.sat[-1])
            rates += rate

        return self.sat, self.x, self.dens, self.dens_m, self.mu, rates, self.kr, self.pc, self.ph

    def evaluate_thermal(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]
        temperature = vec_state_as_np[-1]
        zc = np.append(vec_state_as_np[1:self.nc], 1 - np.sum(vec_state_as_np[1:self.nc]))
        if zc[-1] < 0:
            # print(zc)
            zc = self.comp_out_of_bounds(zc)

        # Enthalpy and conductivity
        for j in self.ph:
            self.enthalpy[j] = self.enthalpy_ev[self.phases_name[j]].evaluate(pressure, temperature, self.x[j][:])  # kJ/kmol
            self.cond[j] = self.conductivity_ev[self.phases_name[j]].evaluate(pressure, temperature, self.x[j][:], self.dens[j])

        # Rock
        rock_energy = self.rock_energy_ev.evaluate(temperature)

        # Heat source and reaction enthalpy (if needed)
        heat_source = 0
        if self.heat_source_ev:
            heat_source += self.heat_source_ev.evaluate(state)

        for j, reaction in enumerate(self.kinetic_rate_ev):
            heat_source += reaction.evaluate_enthalpy(pressure, temperature, self.x, self.sat[-1])

        return self.enthalpy, self.cond, rock_energy, heat_source

    def evaluate_at_cond(self, state):
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        zc = np.append(vec_state_as_np[1:self.nc], 1 - np.sum(vec_state_as_np[1:self.nc]))
        if zc[-1] < 0:
            # print(zc)
            zc = self.comp_out_of_bounds(zc)

        if self.thermal:
            temperature = vec_state_as_np[-1]
        else:
            temperature = self.temperature

        self.sat[:] = 0

        ph = self.run_flash(pressure, temperature, zc)

        for j in ph:
            M = 0
            # molar weight of mixture
            for i in range(self.nc):
                M += self.Mw[i] * self.x[j][i]
            self.dens_m[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, temperature, self.x[j][:]) / M

        self.compute_saturation(ph)

        return self.sat, self.dens_m
