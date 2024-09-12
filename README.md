# GGA-code
Code used for simulations in the framework of the GGA article, that can be find [here](https://arxiv.org/abs/2306.17159)

### General informations

In the folder codes, you should find every code used to produce the simulations in this article. 
A quick description of the files:

+ *adapt_setup.py* contains the technical information to launch a (GGA)Adapt-VQE simulation, for both a molecular system or a Ising 1D chain.
+ *adapt_vqe.py* contains the Adapt-VQE and GGA-VQE protocols. This is the main algorithm file, derived and modified from the corresponding file available in the Qiskit-Algorithms community module.
+ *data_analysis.py* contains a script to process the file created at the end of a whole simulation, to extract the more important information for further analysis.
+ *excitations.py* contains several circuits used to represent excitation operators for the molecular systems and Ising 1D chains.
+ *Ising.py* prepares the required data for doing a simulation for an Ising 1D chain.
+ *main.py* is the main file used to launch the wanted simulation (GGA-VQE or Adapt-VQE) for the targeted system.
+ *observables_evaluator.py* is a modified version of the corresponding file available in the Qiskit-Algorithms community module, specifically to allow shots noises computation.
+ *OperatorPool.py* contains the QEB molecular pool and the minimal Ising pool. 
+ *plot.py* contains a prototypal version of a function to plot a comparison between a GGA-VQE simulation and a native Adapt-VQE one.
+ *UsefulFunctions.py* contains all the relative functions that is at some point used by the other files of this repository.
+ *vqe.py* is a modified version of the corresponding file in the Qiskit-Algorithms community module, specifically to allow a better implementation of shot noise simulations and local (1D) Adapt-VQE optimisation.

### User guide

To use the codes present in this repository, we printed below the ArgParse help for the three codes using them, namely  *main.py*, *data_analysis.py* and *plot.py*.

The options for the *main.py* code:
'''sh

    python main.py --help
    usage: python main.py --help

    options:
    -h, --help            show this help message and exit

    Mandatory Input Options:
    -m SYSTEM_NAME, --system_name SYSTEM_NAME
                            Name of the simulated system: can be either a molecule, e.g. H2, or a Ising chain, e.g. Ising12 for 12 qubits/spins
    -a {adapt,gga}, --algo {adapt,gga}
                            Type of VQE algorithm
    -o {COBYLA,BFGS,POWELL}, --optimizer {COBYLA,BFGS,POWELL}
                            Classical method for the optimization process
    -p {SD,GSD,fermionic,ising_minimal}, --pool_name {SD,GSD,fermionic,ising_minimal}
                            Type of Pool used (SD or GSD)
    -n ADAPT_MAX_ITERATIONS, --adapt_max_iterations ADAPT_MAX_ITERATIONS
                            Max number of ADAPT-VQE interations

    Optional Molecular Options:
    -s SPIN_MULTIPLICITY, --spin_multiplicity SPIN_MULTIPLICITY
                            Value of the spin multiplicity of the molecule, 1 by default (for singlet)
    -c CHARGE, --charge CHARGE
                            Charge of the molecule
    -b BASIS, --basis BASIS
                            Basis set used for the simulation (sto-3g or 6-31G)

    Optional Ising Model Options:
    --coupling COUPLING   Value of the Ising coupling
    --field FIELD         Value of the magnetic field

    Optional Algorithm Options:
    --grad_th GRAD_TH     Value of the gradient threshold
    --eigen_th EIGEN_TH   Value of the eigenvalue threshold (difference between two calculated eigenvalues)
    --energy_th ENERGY_TH
                            Value of the energy threshold between calculated eigenvalue and fci energy
    --method {evolution,circuit}
                            Method of operator's implementation
    --cpu_count CPU_COUNT
                            Number of cpus to use for parallel code
    --restart             Restart file containing ansatz operators and optimal angles

    Noisy Simulation Options:
    --shots SHOTS         Number of shots in the simulation
    --noisy_value {vqe,full}
                            Indicates if full noisy simulation or only vqe noisy for the adapt-vqe simulation
    --reoptimisation      If used, will performed the reoptimisation process for the GGA algorithm
    --adapt_1d            If used, 1D optimisation for ADAPT-VQE procedure

    ADAPT-VQE runner
'''

The options for the *data_analysis.py*:
'''sh 

    python data_analysis.py --help 
    usage: python data_analysis_data.py --help

    options:
    -h, --help            show this help message and exit
    -m MOLECULE_NAME, --molecule_name MOLECULE_NAME
                            Name of the simulated molecule
    -s SHOTS, --shots SHOTS
                            Number of shots
    -p POOL, --pool POOL  Pool used
    -o {all,COBYLA,BFGS,POWELL}, --optimizer {all,COBYLA,BFGS,POWELL}
                            Name of the optimizer
    -a {gga,adapt,gga_reopt}, --algorithm {gga,adapt,gga_reopt}
                            Name of the algorithm used (gga or adapt)
    --run_number RUN_NUMBER
                            Number of the run

    ADAPT data cleaning runner
'''

The options for the *plot.py*:
'''sh 

    python plot.py --help
    usage: python plot.py --help

    options:
    -h, --help            show this help message and exit
    -m MOLECULE_NAME, --molecule_name MOLECULE_NAME
                            Name of the simulated molecule
    -p POOL, --pool POOL  Name of the operator pool used
    -a {noiseless,vqe,full}, --algorithm_method {noiseless,vqe,full}
                            Name of the method used for ADAPTvsGGA: exact, vqe_noisy, full_noisy (vqe + gradient screening)
    -s SHOTS, --shots SHOTS
                            Number of shots used in the simulation
    -e, --error_bar       Indicates if error bar are going to be printed as well
    -o {COBYLA,BFGS,POWELL,all}, --optimizer {COBYLA,BFGS,POWELL,all}
                            Name of the optimizer used for the simulation; if several were used, use all

    ADAPTvsGGA plot runner
'''