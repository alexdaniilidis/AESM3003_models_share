from darts.engines import value_vector, redirect_darts_output, sim_params
import pandas as pd
import matplotlib.pyplot as plt
import os



# check if the output directory exists
if not os.path.exists('Output'):
    # if not, and make it
    os.mkdir('Output/')


# list of application, model and grid options
application = ['GT', 'CCS', 'DO']
model = ['Model1', 'Model2', 'Model3']
grid = ['VeryFine', 'Fine', 'Coarse']

# select application, model, grid and define filename
filename = application[0]+ '_' + model[0] + grid[2]
dir_filename = r'Output/'+filename

# setup log file
redirect_darts_output('%s.log'%dir_filename)

# load application physics model based on filename
if 'GT' in filename:
    from model_geothermal import Model
    plot_cols = ['BHP', 'temperature', 'water rate']
if 'CCS' in filename:
    from model_co2 import Model
    plot_cols = ['BHP', 'wat rate', 'gas rate']
if 'DO' in filename:
    from model_2ph_do import Model
    plot_cols = ['BHP', 'water rate', 'oil rate']


# Load grid file of geo-model at defined resolution
grid_file = filename.split('_')[1]
if '1' in grid_file:
    model_dir = r'Reservoir Models/Arcuate Platform (Gentle Clinoform Dip)/'
if '2' in grid_file:
    model_dir = r'Reservoir Models/Elongated Platform (Steep Clinoform Dip)/'
if '3' in grid_file:
    model_dir = r'Reservoir Models/Elongated Platform (Gentle Clinoform Dip)/'

# construct model
m = Model(model_dir+grid_file+'.grdecl')

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
