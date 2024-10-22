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


def plotit(config_a: dict,data: pd.DataFrame,path: str) -> None:
    """
    The main program that makes the plot(s)
    """

    config_b = copy.deepcopy(config_a)

    expt = config_b["data"]["expt"]
#    var = config_b["data"]["var"]
    init_beg = config_b["data"]["init_beg"]
    init_end = config_b["data"]["init_end"]

    for var in config_b["data"]["var"]:

        patterns = {
                    "expt": expt,
                    "var": var,
                    "init_beg": init_beg,
                    "init_end": init_end,
                }

        print(config_b["data"]["var"])
        print(config_b["data"]["obtype"])
        print(config_b["plot"]["lonrange"])
        print(config_b["plot"]["filename"].format_map(patterns))

        print(config_b["plot"]["title"])



    # Make a Mercator map of the data using Cartopy
    plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.Mercator())
    ax.set_title(var + ' O-F')
    ax.coastlines()
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.LAND, edgecolor='black')
    ax.add_feature(cartopy.feature.LAKES, edgecolor='black')
    ax.add_feature(cartopy.feature.STATES, edgecolor='black')
    ax.gridlines()

    # Plot the air temperature as colored circles and the wind speed as vectors.
    im = plt.scatter(
        data.longitude,
        data.latitude,
        c = data.obs - data.fcst,
        s = 10,
        cmap = "plasma",
        transform = ccrs.PlateCarree(),
    )

    plt.colorbar(im, fraction=0.0263, pad=0.025).set_label(data['unit'].iloc[0])
    plt.tight_layout()
    plt.show()

    # Make sure any subdirectories exist before we try to write the file
    #os.makedirs(os.path.dirname(outfile),exist_ok=True)
    #plt.savefig(outfile)
    #plt.close()


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


    path = '/scratch2/BMC/fv3lam/mayfield/agile_plots/stat_files/point_stat_NSSL-MPAS-HN_mem000_ADPSFC_NDAS_460000L_20240503_100000V.stat'
    var = 'TMP'

    # The data are in a pandas.DataFrame
    #data = vd.datasets.fetch_texas_wind()
    data = pd.read_csv(path, sep='\s+', skiprows=1, header=None, usecols=[1,9,10,15,23,26,27,28,31,32], names=['model','var','unit','type','MPR','station','latitude','longitude','obs','fcst'])

    data = data.loc[data['MPR'] == 'MPR']
    data = data.loc[data['var'] == var]

    print(data)
    print(data.head())


    if os.path.isfile(confg_d["data"]["filename"]):
        files = [confg_d["data"]["filename"]]
    elif glob.glob(confg_d["data"]["filename"]):
        files = sorted(glob.glob(confg_d["data"]["filename"]))
    elif isinstance(confg_d["data"]["filename"], list):
        files = confg_d["data"]["filename"]
    else:
        raise FileNotFoundError(f"Invalid filename(s) specified:\n{confg_d['data']['filename']}")

    for f in files:
        print(f)
        # Open specified file and load dataset
        #dataset=load_dataset(f)


        # Make the plots!
    plotit(confg_d,data,path)