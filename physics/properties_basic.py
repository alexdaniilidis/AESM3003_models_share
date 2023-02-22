import numpy as np
from numba import jit


class ConstProp:
    def __init__(self, value):
        self.value = value

    def evaluate(self, dummy1=0, dummy2=0, dummy3=0, dummy4=0):
        return self.value


# Uncomment these two lines if numba package is installed and make things happen much faster:
@jit(nopython=True)
def RR_func(zc, k, eps):

    a = 1 / (1 - np.max(k)) + eps
    b = 1 / (1 - np.min(k)) - eps

    max_iter = 200  # use enough iterations for V to converge
    for i in range(1, max_iter):
        V = 0.5 * (a + b)
        r = np.sum(zc * (k - 1) / (V * (k - 1) + 1))
        if abs(r) < 1e-12:
            break

        if r > 0:
            a = V
        else:
            b = V

    if i >= max_iter:
        print("Flash warning!!!")

    x = zc / (V * (k - 1) + 1)
    y = k * x

    return (x, y, V)


class Flash:
    def __init__(self, components, ki, min_z=1e-11):
        self.components = components
        self.min_z = min_z
        self.ki = np.array(ki)

    def evaluate(self, pressure, temperature, zc):

        (x, y, V) = self.RR(zc, self.ki)
        return np.array([V, 1-V]), np.array([y, x])

    def RR(self, zc, k):
        return RR_func(zc, k, self.min_z)


class Density:
    def __init__(self, components, dens0=1000, compr=0, p0=1, x_mult=0):
        self.compr = compr
        self.p0 = p0
        self.dens0 = dens0
        self.x_max = x_mult

        if "CO2" in components:
            self.CO2_idx = components.index("CO2")
        else:
            self.CO2_idx = None

    def evaluate(self, pressure, temperature, x):
        if self.CO2_idx:
            x_co2 = x[self.CO2_idx]
        else:
            x_co2 = 0

        density = (self.dens0 + x_co2 * self.x_max) * (1 + self.compr * (pressure - self.p0))
        return density


class ViscosityConst:
    def __init__(self, visc):
        self.visc = visc

    def evaluate(self, dummy1=0, dummy2=0, dummy3=0, dummy4=0):
        return self.visc


class Enthalpy:
    def __init__(self, tref=273.15, hcap=0.0357):
        self.tref = tref
        self.hcap = hcap

    def evaluate(self, temp, dummy1=0, dummy2=0):
        # methane heat capacity
        enthalpy = self.hcap * (temp - self.tref)  # kJ/kmol
        return enthalpy  # kJ/kmol


class PhaseRelPerm:
    def __init__(self, phase, swc=0, sgr=0):
        self.phase = phase

        self.Swc = swc
        self.Sgr = sgr
        if phase == "oil":
            self.kre = 1
            self.sr = self.Swc
            self.sr1 = self.Sgr
            self.n = 2
        elif phase == 'gas':
            self.kre = 1
            self.sr = self.Sgr
            self.sr1 = self.Swc
            self.n = 2
        else:  # water
            self.kre = 1
            self.sr = self.Swc
            self.sr1 = self.Sgr
            self.n = 2

    def evaluate(self, sat):
        if sat >= 1 - self.sr1:
            kr = self.kre
        elif sat <= self.sr:
            kr = 0
        else:
            # general Brook-Corey
            kr = self.kre * ((sat - self.sr) / (1 - self.Sgr - self.Swc)) ** self.n

        return kr


class CapillaryPressure:
    def __init__(self, p_entry=0, swc=0, labda=2):
        self.swc = swc
        self.p_entry = p_entry
        self.labda = labda
        self.eps = 1e-3

    def evaluate(self, sat_w):
        Se = (sat_w - self.swc)/(1 - self.swc)
        if Se < self.eps:
            Se = self.eps
        pc = self.p_entry * Se ** (-1/self.labda)

        Pc = np.array([0, pc], dtype=object)

        return Pc


class RockCompactionEvaluator:
    def __init__(self, pref=1, compres=1.45e-5):
        super().__init__()
        self.Pref = pref
        self.compres = compres

    def evaluate(self, state):
        pressure = state[0]

        return (1.0 + self.compres * (pressure - self.Pref))


class RockEnergyEvaluator:
    def __init__(self):
        super().__init__()

    def evaluate(self, temperature):
        T_ref = 273.15
        # c_vr = 3710  # 1400 J/kg.K * 2650 kg/m3 -> kJ/m3
        c_vr = 1  # 1400 J/kg.K * 2650 kg/m3 -> kJ/m3

        return c_vr * (temperature - T_ref)  # kJ/m3
