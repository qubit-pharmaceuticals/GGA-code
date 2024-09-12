import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import ast, os

def run(plot_config: dict):
    '''
    Graphic representations of the molecular ground-state data obtained after an ADAPT-VQE simulation.
    Args:
        plot_config (dict): dictionnary containing all the necessary information to collect the data that should be visualize.
    '''

    import UsefulFunctions as uf

    fci, hf, down_lim, upper_lim = uf.molecular_specific_constants(plot_config['molecule_name'])

    if plot_config['optimizer'] == 'all':
        optimizers = ['BFGS', 'COBYLA', 'POWELL']
        y
    else:
        optimizers =[plot_config['optimizer']]
    
    for optimizer in optimizers:
        adapt_path = os.path.join(script_dir, f"simulations/data/{plot_config['molecule_name']}/{plot_config['shots']}shots/{plot_config['pool']}_{optimizer}_{plot_config['method']}_adapt.txt")    
        gga_path = os.path.join(script_dir, f"simulations/data/{plot_config['molecule_name']}/{plot_config['shots']}shots/{plot_config['pool']}_{optimizer}_gga.txt")
    
        with open(adapt_path) as file:
            filedata = file.readlines()
        
        y_adapt = [hf]
        v_adapt = [0]
        y_adapt_noiseless = [hf]
        for line in filedata:
            columns = line.split()
            y_adapt.append(ast.literal_eval(columns[0]))
            v_adapt.append(ast.literal_eval(columns[2]))
            y_adapt_noiseless.append(ast.literal_eval(columns[3]))
        
        with open(gga_path) as file:
            filedata = file.readlines()
        
        y_gga = [hf]
        v_gga = [0]
        y_gga_noiseless = [hf]
        for line in filedata:
            columns = line.split()
            y_gga.append(ast.literal_eval(columns[0]))
            v_gga.append(ast.literal_eval(columns[2]))
            y_gga_noiseless(ast.literal_eval(columns[3]))

        if plot_config['shots'] == 'noiseless':
            name = plot_config['molecule_name'] + '_' + optimizer + '_' + plot_config['shots']
        else:
            name = plot_config['molecule_name'] + '_' + optimizer + '_' + str(plot_config['shots']) + '_shots'
        
        uf.plot_comparison_cost(
            y_adapt,
            y_gga,
            y_adapt_noiseless,
            y_gga_noiseless,
            v_adapt,
            v_gga,
            fci,
            down_lim,
            upper_lim,
            name,
            plot_config['method'],
        )      

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        epilog="ADAPTvsGGA plot runner",
        usage="python plot.py --help",
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
        "-p",
        "--pool",
        help="Name of the operator pool used",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-a",
        "--algorithm_method",
        help="Name of the method used for ADAPTvsGGA: exact, vqe_noisy, full_noisy (vqe + gradient screening)",
        required=False,
        default='vqe_noisy',
        choices=['noiseless', 'vqe', 'full'],
    )
    parser.add_argument(
        "-s",
        "--shots",
        help="Number of shots used in the simulation",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-e",
        "--error_bar",
        help="Indicates if error bar are going to be printed as well",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        help="Name of the optimizer used for the simulation; if several were used, use all",
        required=False,
        type=str,
        default='COBYLA',
        choices=['COBYLA', 'BFGS', 'POWELL', 'all'],
    )

    args = parser.parse_args()

    plot_config = {
        "molecule_name" : args.molecule_name,
        "pool"          : args.pool,
        "method"        : args.algorithm_method,
        "shots"         : args.shots,
        "error_bar"     : args.error_bar,
        "optimizer"     : args.optimizer,
    }    

    run(plot_config)