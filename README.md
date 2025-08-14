# GRIPS 2025 Berlin Project: SynLab Matrix Visualization Tool
## Authors: Daniel Hwang (Georgia Institute of Technology), Kathy Mo (University of California, Los Angeles), Sonali Prajapati (Ludwig Maximillian University of Munich)

## Synopsis
The goal of this project is to create a tool that visualizes matrices, more specifically, constraint matrices associated with MIPS, and extracts useful information/patterns from them.

## Tools Used
PyQt to create the graphical interface, PySciPoPT to load in the MIPS model, along with common libraries such as Scipy, Numpy, Matplotlib, Seaborn, etc...
For more details, please see our final report.

## How To Set Up
1. Clone this repo. Ensure Python is at least version 3.11. *If Python is version 3.10, then NetworkX may not fully function with computing the bipartite layout.*
2. Open Terminal (Mac/Linux) or Command Prompt (Windows) and change directory to current repo. One way to do this via GitHub Desktop is to right click on the repo name, click "Copy repo path" and run `cd insert_repo_path_here`. NOTE: For Windows VS Code users, this should work with the built-in VS Code terminal, just make sure that your Terminal is set to cmd and NOT powershell.
3. For first-time use (skip if not needed): Create virtual environment by running `python -m venv env`. Activate virtual environment by running `source env/bin/activate` (Mac) or `env\Scripts\activate.bat` (Windows). Run `pip install -r requirements.txt` to load in all libraries in use. For developers, periodically update libraries used by running `pip freeze > requirements.txt`.

## Files to Run - Viewers
- Run `python mps_matrix_viewer.py` (this will open a matrix visualizaiton tool that visualizes the matrix as a binary scatterplot, a magnitude scatterplot, and a row-scaled heatmap).
- Run `python mps_graph_viewer.py` (this will open a graph visualizaiton tool that visualizes the primal graph, dual graph, and incidence graph associated with the matrix)

## Files to Run - Statistics
- Run `python benchmark_matrix_statistics.py` (requires download of `benchmark.zip` from the MIPLIB website to this folder, script will process each mps file in `benchmark.zip` (restricted to files up to a certain number of constraints and variables), generate basic statistics about each matrix/mps file, and create a matrix statistics csv file called `output/statistics_matrix.csv` - note that `output/` is included in .gitignore, you need to bring this file out of `output/` if you want to save it. Our current version is stored as `statistics_matrix.csv`).
- Run `python benchmark_matrix_viewer_timings.py` (requires download of `benchmark.zip` from the MIPLIB website to this folder, script will process each mps file in `benchmark.zip` (restricted to files up to a certain number of constraints and variables) and time how long it takes each portion of our matrix viewer code, and creates a timing statistics csv file called `timing_statistics.csv`.)

## Developer's Notes: GitHub Desktop workflow.
1. Clone the repo if you haven't already. Remember to fetch origin to make sure your files are synced with the repo!!!
2. Create a new branch based off main - name it whatever you wish. **DO NOT DIRECTLY PUSH TO MAIN.**
3. Make changes, commit them, and push to origin. They should now be available on GitHub.
4. To push changes onto main, open a pull request from your branch into main. If there are manual conflicts, resolve them before proceeding. Your changes should now be updated onto main!
5. **REMEMBER PERIODICALLY TO CLICK CURRENT BRANCH AND MERGE MAIN INTO YOUR BRANCH.** If there are manual conflicts, resolve them before proceeding.

## Developer Note: Visual Studio Code
Our IDE of choice was VS Code, and we used the Python Interpreter plugin in order to run our code. You may also alternatively run the Command Terminal prompts above.

## Acknowledgements
Our mentors this summer were Timo Berthold (FICO Optimization), along with Mohammed Ghannam, Gioni Mexi, and Liding Xu of Zuse Institute Berlin (ZIB). The authors of this repository acknowledge funding from IPAM (US participants) and MODAL (EU participants) for the summer, along with ZIB for their gracious hospitality.