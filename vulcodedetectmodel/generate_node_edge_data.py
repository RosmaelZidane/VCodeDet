# the code here generate_node_edge_data.py, refers to the getgraph.py

import os
import pickle as pkl
import sys
import pandas as pd
import numpy as np
from inits import __inits__important as imp
import data_preprocessing as dpre  ### Always usde this fucntion if you have to use a function from data_preprocessing.py
import graph_subprocess as gsp     # always use this definition if there is a function to use from graph_subprocess.py

# SETUP
NUM_JOBS = 1  
JOB_ARRAY_NUMBER = 0 #if "ipykernel" in sys.argv[0] else int(sys.argv[1]) - 1

# Read Data
df = dpre.datasetss()
# df = df[df['vul'] == 1] # this was to check whether they are funnction with enough change so that we can keep certain processing defined by linevd

df = df.iloc[::-1]
#df = df.transpose
splits = np.array_split(df, NUM_JOBS)


def preprocess(row):
    """Parallelise svdj functions.

    Example:
    df = svdd.datasetss()
    row = df.iloc[180189]  # PAPER EXAMPLE
    row = df.iloc[177860]  # EDGE CASE 1
    preprocess(row)
    """
    savedir_before = imp.get_dir(imp.processed_dir() / row["dataset"] / "before")
    savedir_after = imp.get_dir(imp.processed_dir() / row["dataset"] / "after")

    # Write C Files
    fpath1 = savedir_before / f"{row['id']}.java"
    with open(fpath1, "w") as f:
        f.write(row["before"])
    fpath2 = savedir_after / f"{row['id']}.java"
    if len(row["diff"]) > 0:
        with open(fpath2, "w") as f:
            f.write(row["after"])

    # Run Joern on "before" code
    if not os.path.exists(f"{fpath1}.edges.json"):
        gsp.full_run_joern(fpath1, verbose=3)

    # Run Joern on "after" code
    if not os.path.exists(f"{fpath2}.edges.json") and len(row["diff"]) > 0:
        gsp.full_run_joern(fpath2, verbose=3)

if __name__ == "__main__":
    imp.dfmp(splits[JOB_ARRAY_NUMBER], preprocess, ordr=False, workers=8)