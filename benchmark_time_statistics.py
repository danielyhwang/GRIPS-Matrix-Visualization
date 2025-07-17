# This code tests how fast we can load in a constraint matrix from MPS files using PySciPoPT.

import os
import zipfile
import gzip
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from pyscipopt import Model
from numpy.linalg import matrix_rank
import time
from scipy.sparse import csr_matrix

# Define directories
BASE_DIR = Path(__file__).resolve().parent
BENCHMARK_ZIP = BASE_DIR / "benchmark.zip"
BENCHMARK_DIR = BASE_DIR / "benchmark"
MPS_DIR = BASE_DIR / "mps"
OUTPUT_DIR = BASE_DIR / "output"
TIME_FILE = OUTPUT_DIR / "time_statistics.csv"
time_records = []

# Create necessary directories
MPS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Step 1: Unzip benchmark.zip
print("🔍 Step 1: Unzipping benchmark.zip...")
# Make benchmark directory
try:
    BENCHMARK_DIR.mkdir(exist_ok = False) #If exist_ok is False, creating an existing directory in benchmark
    # will raise a FileExistsError, to which we simply say that the benchmark directory exists. 
    with zipfile.ZipFile(BENCHMARK_ZIP, 'r') as zip_ref:
        zip_ref.extractall(BENCHMARK_DIR)
    print("✅ Unzipped benchmark.zip to ./benchmark")
except FileExistsError:
    print("./benchmark already exists!")

# Step 2: Unpack all .gz files into mps directory
print("🔍 Step 2: Decompressing .gz files in ./benchmark into ./mps folder...")
for gz_file in BENCHMARK_DIR.rglob("*.gz"):
    mps_target = MPS_DIR / gz_file.stem
    print(f"📂 Decompressing: {gz_file.name}")
    if not mps_target.exists():
        with gzip.open(gz_file, 'rb') as f_in:
            with open(mps_target, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        print(f"{gz_file.name} already exists!")
print("✅ Decompressed all .gz files")

# Step 3: Process MPS files
def process_mps(file_path, stem):
    print(f"🔄 Processing MPS file: {stem}")
    output_path = OUTPUT_DIR / stem
    output_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    model = Model()
    model.readProblem(str(file_path))

    variables = model.getVars()
    constraints = model.getConss()

    # Skipping this, we'll see how far we can get!

    #if len(vars_) > 10000 or len(conss) > 10000:
    #    print(f"⚠️ Skipping {stem} due to size > 10,000")
    #    return
    
    var_names = [var.name for var in variables]
    var_index = {name: idx for idx, name in enumerate(var_names)}
    n_vars = len(variables)
    n_cons = len(constraints)
    number_of_nonzeros = 0
    row_inds, col_inds, data = [], [], []
    for i, cons in enumerate(constraints):
        terms = model.getValsLinear(cons)
        for var_name, coef in terms.items():
            j = var_index[var_name]
            if coef != 0:
                number_of_nonzeros += 1
                row_inds.append(i)
                col_inds.append(j)
                data.append(coef)
    
    sparsity = number_of_nonzeros/n_vars/n_cons

    # Read in sparse matrix.
    A_sparse = csr_matrix((data, (row_inds, col_inds)), shape=(n_cons, n_vars))

    end_time = time.time()

    stats = {
        "file": stem,
        "variables": n_vars,
        "constraints": n_cons,
        "sparsity": sparsity,
        "time": end_time - start_time
    }
    time_records.append(stats)

    print(f"✅ Finished processing {stem} in {end_time - start_time} seconds\n")

# Step 4: Loop over all .mps files
print("🔍 Step 3: Extracting constraint matrices and statistics...")
for mps_file in MPS_DIR.glob("*.mps"):
    try:
        process_mps(mps_file, mps_file.stem)
    except Exception as e:
        print(f"❌ Error processing {mps_file.name}: {e}")

# Step 5: Save overall summary
if time_records:
    pd.DataFrame(time_records).to_csv(TIME_FILE, index=False)
    print(f"📊 Time summary written to {TIME_FILE}")

print("🎉 All benchmark files processed successfully!")
