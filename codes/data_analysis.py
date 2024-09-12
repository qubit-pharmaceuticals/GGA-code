import csv
import os
import ast
import json
import sys

import numpy as np
from scipy.interpolate import CubicSpline, interp1d

csv.field_size_limit(sys.maxsize)

def run(data_config):
    '''
    Runs a specific analysis of the data obtained from a (GGA)Adapt-VQE simulation for a molecular system. 
    Processes this data to produce several files for a qualitative analysis of these data.
    Args:
        data_config (dict): dictionnary containing all the necessary information to retrieve the correct simulated data and to produce the corresponding needed optimal files. 
    '''

    #Setting up the possible optimizers used in the simulations
    if data_config['optimizer'] == 'all':
        optimizers = ['BFGS', 'COBYLA', 'POWELL']
    else:
        optimizers = [data_config['optimizer']]
    
    ###Loop on each possible optimizer used (made to be used for analyzing data for runs with the different allowed classical optimizers)##
    for optimizer in optimizers:
        #Creating all the needed paths to treat the data
        script_dir = os.path.dirname(__file__)
        if data_config['algorithm'] == 'gga' or data_config['algorithm'] == 'gga_reopt':
            path = os.path.join(script_dir, f"simulations/data/{data_config['molecule_name']}/{data_config['shots']}/run{data_config['run_number']}/{data_config['pool']}_{optimizer}_{data_config['algorithm']}.csv") 
            op_path = os.path.join(script_dir, f"simulations/data/{data_config['molecule_name']}/{data_config['shots']}/run{data_config['run_number']}/op_{data_config['algorithm']}.csv")
        elif data_config['algorithm'] == 'adapt':
            path = os.path.join(script_dir, f"simulations/data/{data_config['molecule_name']}/{data_config['shots']}/run{data_config['run_number']}/{data_config['pool']}_{optimizer}_full_adapt.csv")
        temp_path = os.path.join(script_dir, f"simulations/data/{data_config['molecule_name']}/{data_config['shots']}/run{data_config['run_number']}/temp.csv")

        ##Eliminating all non necessary lines and lines that cannot be used with pythonic strings' conversion, through a temporary file##
        with open(path, "r") as input:
            with open(temp_path, "w") as output:
                for line in input:
                    if data_config['algorithm'] == 'gga' or data_config['algorithm'] == 'gga_reopt':
                        if "SparsePauliOp" not in line.strip("\n") and "PauliSumOp" not in line.strip("\n") and "coeff" not in line.strip("\n") and "0.j" not in line.strip("\n"):
                            output.write(line)                            
                    elif data_config['algorithm'] == 'adapt':
                        if "optimal_circuit" not in line.strip("\n") and "ParameterVectorElement" not in line.strip("\n") and "optimizer_result" not in line.strip("\n") and "Parameter" not in line.strip("\n"):
                            output.write(line)
        
        ##Obtaining the operators used to create the ansatz##
        if data_config['algorithm'] == 'gga' or data_config['algorithm'] == 'gga_reopt':
            with open(path, "r") as input:
                with open(op_path, "w") as output:
                    for line in input:
                        if "SparsePauliOp" in line.strip("\n"):
                            output.write(line)
        
        with open(temp_path, "r") as input:
            filedata = input.read()
        
        filedata = filedata.replace("array", "")
        filedata = filedata.replace("'termination_criterion': <TerminationCriterion.MAXIMUM: 'Maximum number of iterations reached'>", "")
        filedata = filedata.replace("'termination_criterion': <TerminationCriterion.CONVERGED: 'Threshold converged'>", "")

        with open(temp_path, "w") as output:
            output.write(filedata)
        
        ##Getting the wanted data from the previously created temporary file temp.csv##
        rows = []
        with open(temp_path) as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader: 
                rows.append(row)
        
        #Getting the main metadata dictionnary (containing optimal parameters, optimal and noiseless energies, associated variances, operators, ...)
        dict_metadata = eval(rows[0][0])

        #Getting the numbers of cost function evaluations for the native ADAPT workflow
        if data_config['algorithm'] == 'adapt':
            cost_fun_evals = json.loads(rows[2][0])
        
        #Creating the useful lists in which will be stored all the data for both ADAPT and GGA-ADAPT
        if data_config['shots'] == 'noiseless':
            noiseless_energies = []
            if data_config['algorithm'] == 'gga' or data_config['algorithm'] == 'gga_reopt':
                mean_variances = []
                multi_variances = []
            elif data_config['algorithm'] == 'adapt':
                metadata_variances = []
        else:
            if data_config['algorithm'] == 'gga' or data_config['algorithm'] == 'gga_reopt':
                multi_variances = eval(rows[2][0])
                mean_variances = eval(rows[4][0])
                reopti_energies = eval(rows[8][0])
                reopti_angles = eval(rows[10][0])
            elif data_config['algorithm'] == 'adapt':
                metadata_variances = eval(rows[4][0])
            noiseless_energies = eval(rows[6][0])
        
        ##Storing in the associated lists the energies, optimal angles and simulations' variances##
        if data_config['algorithm'] == 'adapt':
            #Getting eigenvalues and associated variances from simulations
            eigenvalues = dict_metadata['eigenvalue_history']
            variances = []
            for i in range(len(metadata_variances)):
                variances.append(metadata_variances[i]['variance'])
        elif data_config['algorithm'] == 'gga' or data_config['algorithm'] == 'gga_reopt':
            #Getting eigenvalues and optimal angles from simulations
            eigenvalues = []
            opt_angles = []
            for i in range(len(dict_metadata['eigenvalue_history'])):
                eigenvalues.append(dict_metadata['eigenvalue_history'][i][0])
                opt_angles.append(dict_metadata['eigenvalue_history'][i][1][0])
            angles = [0, np.pi, np.pi/2, -np.pi/2, np.pi/3]
            #Calculating interpolated variances for the optimal angles
            interp_variances = []
            for i in range(len(multi_variances)):
                new_data = [multi_variances[i][3], multi_variances[i][0], multi_variances[i][4], multi_variances[i][2], multi_variances[i][1]]
                interp = interp1d(sorted(angles), new_data, fill_value="extrapolate")
                interp_variances.append(interp(opt_angles[i]))
            #Generating the lists storing iterative angles and energies for the 1D reoptimization of GGA
            op_rows = []
            with open(op_path) as opcsvfile:
                opcsvreader = csv.reader(opcsvfile)
                for row in opcsvreader:
                    op_rows.append(row)
            
            ops = []
            for i in range(len(op_rows)):
                string_exci = ""
                for j in range(2, len(op_rows[i][-2])-2):
                    string_exci += op_rows[i][-2][j]
                name_op = ""
                for j in range(len(string_exci)):
                    if string_exci[j] == "X" or string_exci[j] == "Y":
                        name_op += str(j) + "_"
                ops.append(name_op)
        
        #Creating a numbered list for iteration
        iterations = []
        for i in range(1,51):
            iterations.append(i)

        ##Generating the output files with all of the necessary data for exact and shot-noisy simulations##
        if data_config['shots'] == 'noiseless':
            if data_config['algorithm'] == 'gga' or data_config['algorithm'] == 'gga_reopt':
                data_path = os.path.join(script_dir, f"simulations/data/{data_config['molecule_name']}/noiseless/run{data_config['run_number']}/{data_config['pool']}_{optimizer}_{data_config['algorithm']}.txt")
                with open(data_path, 'w') as file:
                    title_list = [["Iteration"], ["Operator"], ["Energy"]]
                    for x in zip(*title_list):
                        file.write("{0}\t{1}\t{2}\n".format(*x))
                    lst = [iterations, ops, eigenvalues]
                    for x in zip(*lst):
                        file.write("{0}\t{1}\t{2}\n".format(*x))
            elif data_config['algorithm'] == 'adapt':
                data_path = os.path.join(script_dir, f"simulations/data/{data_config['molecule_name']}/noiseless/run{data_config['run_number']}/{data_config['pool']}_{optimizer}_adapt.txt")
                with open(data_path, 'w') as file:
                    title_list = [["Iteration"], ["Operator"], ["Energy"], ["Cost function evaluations"]]
                    for x in zip(*title_list):
                        file.write("{0}\t{1}\t{2}\t{3}\n".format(*x))
                    lst = [iterations, ops, eigenvalues, cost_fun_evals]
                    for x in zip(*lst):
                        file.write("{0}\t{1}\n".format(*x))
        else:
            if data_config['algorithm'] == 'gga' or data_config['algorithm'] == 'gga_reopt':
                data_path = os.path.join(script_dir, f"simulations/data/{data_config['molecule_name']}/{data_config['shots']}/run{data_config['run_number']}/{data_config['pool']}_{optimizer}_{data_config['algorithm']}.txt")
                with open(data_path, 'w') as file:
                    title_list = [["Iteration"], ["Operator"], ["Noisy Energy"], ["Mean Variance"], ["Interpolated Variance"], ["Noiseless Energy"]]
                    for x in zip(*title_list):
                        file.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(*x))
                    lst = [iterations, ops, eigenvalues, mean_variances, interp_variances, noiseless_energies]
                    for x in zip(*lst):
                        file.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(*x))
            elif data_config['algorithm'] == 'adapt':
                data_path = os.path.join(script_dir, f"simulations/data/{data_config['molecule_name']}/{data_config['shots']}/run{data_config['run_number']}/{data_config['pool']}_{optimizer}_full_adapt.txt")
                with open(data_path, 'w') as file:
                    title_list = [["Iteration"], ["Noisy Energy"], ["Cost Function evals"], ["Variances"], ["Noiseless Energy"]]
                    for x in zip(*title_list):
                        file.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(*x))
                    lst = [iterations, eigenvalues, cost_fun_evals, variances, noiseless_energies]
                    for x in zip(*lst):
                        file.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(*x))

        if data_config['algorithm'] == 'gga_reopt':
        ##GGA reoptimized iterative energies##
            reopti_data_path = os.path.join(script_dir, f"simulations/data/{data_config['molecule_name']}/{data_config['shots']}/run{data_config['run_number']}/{data_config['pool']}_{optimizer}_gga_reopt_energies.txt")
            with open(reopti_data_path, "w") as file:
                lst = [iterations, reopti_energies]
                for x in zip(*lst):
                    file.write("{0}\t{1}\n".format(*x))

        ##GGA reoptimized iterative angles##
            reopti_data_path = os.path.join(script_dir, f"simulations/data/{data_config['molecule_name']}/{data_config['shots']}/run{data_config['run_number']}/{data_config['pool']}_{optimizer}_gga_reopt_angles.txt")
            with open(reopti_data_path, "w") as file:
                lst = [iterations, reopti_angles]
                for x in zip(*lst):
                    file.write("{0}\t{1}\n".format(*x))

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        epilog="ADAPT data cleaning runner",
        usage="python data_analysis_data.py --help",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-m",
        "--molecule_name",
        help="Name of the simulated molecule",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--shots",
        help="Number of shots",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--pool",
        help="Pool used",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        help="Name of the optimizer",
        required=True,
        type=str,
        choices=['all', 'COBYLA', 'BFGS', 'POWELL']
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        help="Name of the algorithm used (gga or adapt)",
        required=True,
        type=str,
        choices=['gga', 'adapt', 'gga_reopt'],
    )
    parser.add_argument(
        "--run_number",
        help="Number of the run",
        required=True,
        type=int,
    )

    args = parser.parse_args()

    data_config = {
        "molecule_name"  : args.molecule_name,
        "shots"          : args.shots,
        "pool"           : args.pool,
        "optimizer"      : args.optimizer,
        "algorithm"      : args.algorithm,
        "run_number"     : args.run_number,
    }

    run(data_config)