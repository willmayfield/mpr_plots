#!/usr/bin/env python3
"""
Script for plotting residual values graphically from MPR pairs in MET stat output
"""
import argparse
import copy
import glob
import logging
import os
import numpy as np
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import uwtools.api.config as uwconfig 
import cartopy


#def plotit(config_a: dict,data: pd.DataFrame,path: str) -> None:
def plotit(confg_d: dict) -> None:
    """
    The main program that makes the plot(s)
    """

    config_b = copy.deepcopy(confg_d)


    expt = config_b["data"]["expt"]
    init_beg = config_b["data"]["init_beg"]
    init_end = config_b["data"]["init_end"]
    obtype = config_b["data"]["obtype"]
    agg_type = config_b["plot"]["agg_type"]

    filepatterns = {
            "expt": expt,
            "init_beg": init_beg,
            "init_end": init_end,
            "obtype": obtype,
        }

    filenames=config_b["data"]["filename"].format_map(filepatterns)

    print('Filename globbing: '+filenames)

    if os.path.isfile(filenames):
        files = [filenames]
    elif glob.glob(filenames):
        files = sorted(glob.glob(filenames))
    elif isinstance(filenames, list):
        files = filenames
    else:
        #raise FileNotFoundError(f"Invalid filename(s) specified:\n{confg_d['data']['filename']}")
        files = [filenames]
        print(filenames)
        print("file is not right")


    data = []

    for f in files:
        print("Loading file " + f)
        # Open specified file and load dataset
        #dataset=load_dataset(f)
        path = f

    #path = '/scratch2/BMC/fv3lam/mayfield/agile_plots/stat_files/point_stat_NSSL-MPAS-HN_mem000_ADPSFC_NDAS_460000L_20240503_100000V.stat'

        # The data are in a pandas.DataFrame
        data_f = pd.read_csv(path, sep='\s+', skiprows=1, header=None, usecols=[1,9,10,15,23,26,27,28,31,32], names=['model','var','unit','type','MPR','station','latitude','longitude','obs','fcst'])

        data.append(data_f.loc[data_f['MPR'] == 'MPR'])


    data = pd.concat(data, ignore_index=True)

    for var in config_b["data"]["var"]:

        patterns = {
                    "var": var,
                    "expt": expt,
                    "init_beg": init_beg,
                    "init_end": init_end,
                    "obtype": obtype,
                }

        # Select the data from the current variable

        print("Aggregating data from "+str(len(files))+" files with "+str(len(data.loc[data['var'] == var]))+" MPR lines for variable "+var+" using "+agg_type+".")

        data_agg = agg(data.loc[data['var'] == var],agg_type)

        print("Plotting "+str(len(data_agg))+" unique stations.")

        print(config_b["plot"]["title"].format_map(patterns))

        outfile = config_b["plot"]["filename"].format_map(patterns)


    cm_config = config_b["plot"]["colormap"]

    # Make a Mercator map of the data using Cartopy
    plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.Mercator())
    ax.set_title('F-O\n'+expt + '\n' + var + ', 060000L, ' + str(init_beg)+"-"+str(init_end))
    ax.coastlines()
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.LAND, edgecolor='black')
    ax.add_feature(cartopy.feature.LAKES, edgecolor='black')
    ax.add_feature(cartopy.feature.STATES, edgecolor='black')
    ax.gridlines()

    # Plot the air temperature residuals as colored circles.
    im = plt.scatter(
        data_agg.longitude,
        data_agg.latitude,
        c = data_agg.fmo,
        s = 10,
        cmap = cm_config,
    #    clim = (-2.5*np.std(data_agg.fmo),2.5*np.std(data_agg.fmo)),
        clim = (-2.5,2.5),
        transform = ccrs.PlateCarree(),
    )

    plt.colorbar(im, fraction=0.02637, pad=0.015).set_label(data_agg['unit'].iloc[0])
    plt.tight_layout()
    #plt.show()

    # Make sure any subdirectories exist before we try to write the file
    os.makedirs(os.path.dirname(outfile),exist_ok=True)
    plt.savefig(outfile)
    plt.close()

def agg(data: pd.DataFrame, agg_type = "mean"):
    # Caluclate Forecast minus Obs
    data['fmo'] = data['fcst'] - data['obs']

    print(data)
    
    # Aggregate the resulting column along the stations
    if agg_type == "mean":
        return data.groupby('station').agg({'fmo':'mean', 'unit':'first', 'latitude':'first','longitude':'first'})
    elif agg_type == "median":
        return data.groupby('station').agg({'fmo':'median', 'unit':'first', 'latitude':'first','longitude':'first'})
    else:
        print("Invalid aggregation type")

def setup_logging(logfile: str = "log.generate_FV3LAM_wflow", debug: bool = False) -> logging.Logger:
    """
    Sets up logging, printing high-priority (INFO and higher) messages to screen, and printing all
    messages with detailed timing and routine info in the specified text file.

    If debug = True, print all messages to both screen and log file.
    """
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)

    # Create handlers
    console = logging.StreamHandler()
    fh = logging.FileHandler(logfile)

    # Set the log level for each handler
    if debug:
        console.setLevel(logging.DEBUG)
    else:
        console.setLevel(logging.INFO)
    fh.setLevel(logging.DEBUG)  # Log DEBUG and above to the file

    formatter = logging.Formatter("%(name)-22s %(levelname)-8s %(message)s")

    # Set format for file handler
    fh = logging.FileHandler(logfile, mode='w')
    fh.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console)
    logger.addHandler(fh)

    logger.debug("Logging set up successfully")

    return logger

def setup_config(config: str) -> dict:
    """
    Function for reading in dictionary of configuration settings, and performing basic checks
    on those settings
    """
    logging.debug(f"Reading options file {config}")
    try:
        config_d = uwconfig.get_yaml_config(config=config)
    except Exception as e:
        logging.critical(e)
        logging.critical(f"Error reading {config}, check above error trace for details")
        sys.exit(1)
    if not config_d["data"].get("lev"):
        logging.debug("Level not specified in config, will use level 0 if multiple found")
        config_d["data"]["lev"]=0

    logging.debug("Expanding references to other variables and Jinja templates")
    config_d.dereference()
    return config_d

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Script for plotting MPAS input and/or output in native NetCDF format"
    )
    parser.add_argument('-c', '--config', type=str, default='plot_options.yaml',
                        help='File used to specify plotting options')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Script will be run in debug mode with more verbose output')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)


    # Load settings from config file
    confg_d=setup_config(args.config)




        # Make the plots!
    plotit(confg_d)
