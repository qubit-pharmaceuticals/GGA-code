# GGA-code
Code used for simulations in the framework of the GGA article, that can be find [here](https://arxiv.org/abs/2306.17159)

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
