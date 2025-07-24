import sys
import os
import io
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from pyscipopt import Model as SCIPModel
from pygcgopt import Model as GCGModel
from collections import defaultdict


def extract_sparse_matrix(mps_file):
    model = SCIPModel()
    model.readProblem(mps_file)

    vars = model.getVars()
    conss = model.getConss()

    var_names = [v.name for v in vars]
    var_index = {name: i for i, name in enumerate(var_names)}

    row_inds, col_inds, data = [], [], []
    for i, cons in enumerate(conss):
        terms = model.getValsLinear(cons)
        for var_name, coef in terms.items():
            j = var_index[var_name]
            row_inds.append(i)
            col_inds.append(j)
            data.append(coef)

    A = csr_matrix((data, (row_inds, col_inds)), shape=(len(conss), len(vars)))
    return A, [c.name for c in conss], var_names


def save_spy_plot(matrix, filename, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.spy(matrix, markersize=0.5)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def export_dec_file(path, block_row_map, block_col_map):
    block_rows = defaultdict(list)
    block_cols = defaultdict(list)

    for row, block in block_row_map.items():
        block_rows[block].append(row)
    for col, block in block_col_map.items():
        block_cols[block].append(col)

    with open(path, 'w') as f:
        for block in sorted(block_rows):
            f.write(f"BLOCK {block}\n")
            f.write("ROWS\n")
            for row in block_rows[block]:
                f.write(f"{row}\n")
            f.write("COLUMNS\n")
            for col in block_cols[block]:
                f.write(f"{col}\n")


def get_block_assignments(gcg_model):
    block_row_map = {}
    block_col_map = {}

    for cons in gcg_model.getConss():
        block = gcg_model.getBlockCons(cons)
        if block >= 0:
            block_row_map[cons.name] = block

    for var in gcg_model.getVars():
        block = gcg_model.getBlockVar(var)
        if block >= 0:
            block_col_map[var.name] = block

    return block_row_map, block_col_map


def get_permuted_matrix(A, row_perm, col_perm):
    return A[row_perm, :][:, col_perm]


def main(mps_file):
    base = os.path.splitext(mps_file)[0]

    # Step 1: Extract sparse matrix
    A, row_names, col_names = extract_sparse_matrix(mps_file)
    save_spy_plot(A, base + "_original.png", "Original Sparsity Pattern")

    # Step 2: Use GCG for decomposition
    gcg_model = GCGModel(mps_file)
    gcg_model.setIntParam("detection/maxrounds", 10)
    gcg_model.detect()

    # Step 3: Extract block info and export .dec
    block_row_map, block_col_map = get_block_assignments(gcg_model)
    dec_file = base + ".dec"
    export_dec_file(dec_file, block_row_map, block_col_map)
    print(f"Saved decomposition to: {dec_file}")

    # Step 4: Reload model with .dec file and get permutation
    gcg_model = GCGModel()
    gcg_model.readProblem(mps_file)
    gcg_model.readDecomposition(dec_file)
    gcg_model.detect()
    decomp_data = gcg_model.getDetProbData()
    row_perm = decomp_data.getRowsPerm()
    col_perm = decomp_data.getColsPerm()

    # Step 5: Plot permuted matrix
    A_perm = get_permuted_matrix(A, row_perm, col_perm)
    save_spy_plot(A_perm, base + "_decomposed.png", "Decomposed structure of original problem")
    print(f"Saved reordered plot to: {base}_decomposed.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps", required=True, help="Path to MPS file")
    args = parser.parse_args()
    main(args.mps)
