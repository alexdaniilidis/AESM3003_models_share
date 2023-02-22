from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from RRMParser import RRMParser
import numpy as np
from darts.tools.keyword_file_tools import load_single_keyword, save_few_keywords


class RRMBaseModel(DartsModel):
    def __init__(self, n_points=1000,filename='',DX=0.0,DY=0.0,DZ=0.0,DEPTH=0.0,
                 well_dir = ):
        # call base class constructor
        super().__init__()
        self.n_points = n_points
        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        parser = RRMParser(filename)

        # create reservoir
        self.permx = parser.ReturnData('PERMX')
        # self.permx = load_single_keyword(filename, 'PERMX', cache=True)

        self.permy = parser.ReturnData('PERMY')
        self.permz = parser.ReturnData('PERMZ')
        self.poro  = parser.ReturnData('PORO')


        self.actnum = parser.ReturnData('ACTNUM')
        self.coord  = parser.ReturnData('COORD')
        self.zcorn  = parser.ReturnData('ZCORN')

        self.dx = DX
        self.dy = DY
        self.dz = DZ

        NXNYNZ = parser.ReturnNXYNYZ()
        NX = NXNYNZ[0]
        NY = NXNYNZ[1]
        NZ = NXNYNZ[2]

        depths = [DEPTH] * (NX*NY*NZ)
        self.depth = np.array(depths)

        self.reservoir = StructReservoir(self.timer, nx=NX, ny=NY, nz=NZ, dx=self.dx, dy=self.dy, dz=self.dz,
                                         permx=self.permx, permy=self.permy, permz=self.permz, poro=self.poro,
                                         depth=self.depth, actnum=self.actnum, coord=self.coord, zcorn=self.zcorn,
                                         is_cpg=False)

        poro = np.array(self.reservoir.mesh.poro, copy=False)
        poro[poro == 0.0] = 1.E-4
        well_dia = 0.152
        well_rad = well_dia / 2


        keep_reading = True
        prev_well_name = ''
        with open('WELLS.INC') as f:
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


        self.timer.node["initialization"].stop()

    def wells4ParaView(self):
        name = []
        type = []
        ix = []
        iy = []
        keep_reading = True
        with open('WELLS.INC') as f:
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

        f = open('well_gen.txt', 'w')
        str2file(f, 'well_list', name)
        str2file(f, 'well_type', type)
        num2file(f, 'well_x', ix)
        num2file(f, 'well_y', iy)
        f.close()
        print('done')