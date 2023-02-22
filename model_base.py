from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
import numpy as np
from darts.tools.keyword_file_tools import load_single_keyword, save_few_keywords
import os


class BaseModel(DartsModel):
    def __init__(self, n_points=1000, temperature=None):
        # call base class constructor
        super().__init__()
        self.n_points = n_points
        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        if temperature is None:
            self.thermal = True
        else:
            self.thermal = False

        # create reservoir from UNISIM - 20 layers (81*58*20, Corner-point grid)
        self.permx = load_single_keyword(r'Petrel_model/well/reservoir.GRDECL', 'PERMX')
        self.permy = load_single_keyword(r'Petrel_model/well/reservoir.GRDECL', 'PERMX')
        self.permz = load_single_keyword(r'Petrel_model/well/reservoir.GRDECL', 'PERMZ')
        self.poro = load_single_keyword(r'Petrel_model/well/reservoir.GRDECL', 'PORO')
        self.depth = load_single_keyword(r'Petrel_model/well/reservoir.GRDECL', 'DEPTH')

        if os.path.exists(('Petrel_model')):
            print('Reading dx, dy and dz specifications...')
            self.dx = load_single_keyword(r'Petrel_model/well/width.GRDECL', 'DX')
            self.dy = load_single_keyword(r'Petrel_model/well/width.GRDECL', 'DY')
            self.dz = load_single_keyword(r'Petrel_model/well/width.GRDECL', 'DZ')
        else:
            self.dx = self.dy = self.dz = 0

        # Import other properties from files
        filename = 'Petrel_model/well/grid.GRDECL'
        self.actnum = load_single_keyword(filename, 'ACTNUM')

        is_CPG = False  # True for re-calculation of dx, dy and dz from CPG grid
        if is_CPG:
            self.coord = load_single_keyword(filename, 'COORD')
            self.zcorn = load_single_keyword(filename, 'ZCORN')
        else:
            self.coord = self.zcorn = 0

        self.reservoir = StructReservoir(self.timer, nx=100, ny=96, nz=50, dx=self.dx, dy=self.dy, dz=self.dz,
                                         permx=self.permx, permy=self.permy, permz=self.permz, poro=self.poro,
                                         depth=self.depth, actnum=self.actnum, coord=self.coord, zcorn=self.zcorn,
                                         is_cpg=is_CPG)
        self.nb = self.reservoir.nb

        poro = np.array(self.reservoir.mesh.poro, copy=False)
        poro[poro == 0.0] = 1.E-4

        self.reservoir.set_boundary_volume(yz_minus=1e15, yz_plus=1e15, xz_minus=1e15,
                                           xz_plus=1e15, xy_minus=1e15, xy_plus=1e15)

        if is_CPG:
            dx, dy, dz = self.reservoir.get_cell_cpg_widths()
            save_few_keywords('width.in', ['DX', 'DY', 'DZ'], [dx, dy, dz])

        self.read_and_add_perforations('Petrel_model/well/WELLS.INC')

        self.timer.node["initialization"].stop()

    def set_initial_conditions(self):
        mesh = self.reservoir.mesh
        """ Uniform Initial conditions """
        # set initial pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure.fill(self.pressure_ini)

        if self.thermal:
            temperature = np.array(mesh.temperature, copy=False)
            temperature.fill(self.temp_ini)

        nc = self.property_container.nc
        nb = mesh.n_blocks
        mesh.composition.resize(nb * (nc - 1))
        # set initial composition
        composition = np.array(mesh.composition, copy=False)
        for c in range(nc - 1):
            composition[c::(nc - 1)] = self.ini_comp[c]

    def read_and_add_perforations(self, filename):
        well_dia = 0.152
        well_rad = well_dia / 2

        keep_reading = True
        prev_well_name = ''
        with open(filename) as f:
            while keep_reading:
                buff = f.readline()
                if 'COMPDAT' in buff:
                    while True:  # be careful here
                        buff = f.readline()
                        if len(buff) != 0:
                            CompDat = buff.split()

                            if len(CompDat) != 0 and '/' != CompDat[0]:  # skip the empty line and '/' line
                                # define well
                                if CompDat[0] == prev_well_name:
                                    pass
                                else:
                                    self.reservoir.add_well(CompDat[0], wellbore_diameter=well_dia)
                                    prev_well_name = CompDat[0]

                                # define perforation
                                for i in range(int(CompDat[3]), int(CompDat[4]) + 1):
                                    self.reservoir.add_perforation(self.reservoir.wells[-1],
                                                                   int(CompDat[1]), int(CompDat[2]), i,
                                                                   well_radius=well_rad,
                                                                   multi_segment=False)

                            if len(CompDat) != 0 and '/' == CompDat[0]:
                                keep_reading = False
                                break

    def wells4ParaView(self, filename):
        name = []
        type = []
        ix = []
        iy = []
        keep_reading = True
        with open('Petrel_model/well/WELLS.INC') as f:
            while keep_reading:
                buff = f.readline()
                if 'WELSPECS' in buff:
                    while True:  # be careful here
                        buff = f.readline()
                        if len(buff) != 0:
                            welspecs = buff.split()

                            if len(welspecs) != 0 and welspecs[0] != '/' and welspecs[0][:2] != '--':  # skip the empty line and '/' line
                                name += [welspecs[0]]
                                if 'GROUP1' in welspecs[1]:
                                    type += ['PRD']
                                else:
                                    type += ['INJ']
                                ix += [welspecs[2]]
                                iy += [welspecs[3]]
                                # define perforation

                            if len(welspecs) != 0 and welspecs[0] == '/':
                                keep_reading = False
                                break
        f.close()

        def str2file(fp, name_in, list_in):
            fp.write("%s = [" % name_in)
            for item in list_in:
                fp.write("\'%s\', " % item)
            fp.write("]\n")

        def num2file(fp, name_in, list_in):
            fp.write("%s = [" % name_in)
            for item in list_in:
                fp.write("%d, " % int(item))
            fp.write("]\n")

        f = open(filename, 'w')
        str2file(f, 'well_list', name)
        str2file(f, 'well_type', type)
        num2file(f, 'well_x', ix)
        num2file(f, 'well_y', iy)
        f.close()
