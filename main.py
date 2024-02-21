from darts.engines import value_vector, redirect_darts_output, sim_params
import pandas as pd
import matplotlib.pyplot as plt
import os


# check if the output directory exists
if not os.path.exists('Output'):
    # if not, make it
    os.mkdir('Output/')


# list of application, model and grid options
application = ['GT', 'CCS']
model = ['ScenA', 'ScenB']
# grid = ['VeryFine', 'Fine', 'Coarse']

### -> select application <- ####, model, grid and define filename
filename = application[0]+ '_' + model[1] + 'Fine'
dir_filename = r'Output/'+filename

# setup log file
redirect_darts_output('%s.log'%dir_filename)

# load application physics model based on filename
if 'GT' in filename:
    from model_geothermal import Model
    plot_cols = ['BHP', 'temperature', 'water rate']
if 'CCS' in filename:
    from model_co2 import Model
    plot_cols = ['BHP', 'gas rate', 'wat rate']



# Load grid file of geo-model at defined resolution
grid_file = filename.split('_')[1]
if 'A' in grid_file:
    model_dir = r'Reservoir Models/Arcuate Platform (Gentle Clinoform Dip)/'
if 'B' in grid_file:
    model_dir = r'Reservoir Models/Elongated Platform (Steep Clinoform Dip)/'


# construct model
m = Model(model_dir+grid_file+'.grdecl')


# define well diameter and radius
well_dia = 0.152
well_rad = well_dia / 2

# add injector well
m.reservoir.add_well('INJ1')

# add perforations ##### -> HERE YOU CAN CHANGE THE X,Y GRID COORDINATES OF THE WELL <- ####
for k in range(m.reservoir.nz):
    m.reservoir.add_perforation(m.reservoir.wells[0], 25, 10, k+1, well_radius=well_rad,
                                multi_segment=False, verbose=False)

if 'GT' in filename:
    # add production well - CONDITIONAL TO APPLICATION TYPE
    m.reservoir.add_well('PRD1')

    # add perforations ##### -> HERE YOU CAN CHANGE THE X,Y GRID COORDINATES OF THE WELL <- ####
    for k in range(m.reservoir.nz):
        m.reservoir.add_perforation(m.reservoir.wells[1], 25, 50, k+1, well_radius=well_rad,
                                    multi_segment=False, verbose=False)


# Initialise model
m.init()

# Save initial model to VTK file
m.export_pro_vtk(file_name=filename)

# Define run time and execute simulation
days = 365
if 'CCS' in filename:
    years = 10
else:
    years = 50
run_time = days*years
for a in range(years):
    m.run_python(days)
    m.export_pro_vtk(file_name=filename)

m.export_pro_vtk(file_name=filename)

# Write runtime stats
m.print_timers()
m.print_stat()

# get well time_data and save to excel and pickle
time_data = pd.DataFrame.from_dict(m.physics.engine.time_data)
time_data['Time (yrs)'] = time_data['time'] / 365

#remove not needed columns from time_data results
press_gridcells = time_data.filter(like='reservoir').columns.tolist()
chem_cols = time_data.filter(like='kmol').columns.tolist()

# remove collumns from data
time_data.drop(columns=press_gridcells + chem_cols, inplace=True)


time_data.to_excel(dir_filename+'_time_data'+ '.xlsx', 'Sheet1')
time_data.to_pickle(dir_filename+'_time_data'+ '.pkl')



## Do a basic plot to check output

# get well names
well_names = [w.name for w in m.reservoir.wells]

# define figure and get axes
fig, ax = plt.subplots(1, len(plot_cols), figsize=(12,5))
axx = fig.axes

# plot the defined columns for all wells
for i, col in enumerate(plot_cols):
    time_data.plot(x='Time (yrs)', y=time_data.filter(like=col).columns.to_list(), ax=axx[i])
    axx[i].set_ylabel('%s %s'%(col, time_data.filter(like=col).columns.tolist()[0].split(' ')[-1]))
    axx[i].legend(labels=[lab.split(':')[0].split('(')[0] for lab in axx[i].get_legend_handles_labels()[1]],
                      frameon=False, ncol=2)
    axx[i].tick_params(axis=u'both', which=u'both',length=0)
    for location in ['top','bottom','left','right']:
        axx[i].spines[location].set_linewidth(0)
    axx[i].grid(alpha=0.3)
plt.tight_layout()
plt.show()
