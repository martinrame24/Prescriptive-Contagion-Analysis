# Prescriptive-Contagion-Analytics

Dropbox link for the necessary data: https://www.dropbox.com/sh/etl8eosoo1vmijg/AADQAz3gaHFKMW7EiGTm7u9Ta?dl=0

Regarding code files, the folder SIR Fitting contains the code to get the model used for traffic congestion. 

The Branch-and-Price folder contains all the necessary code for running the BP on each use case detailed in the paper.\\
The file global_constants.jl defines some global constants for the BP to run properly. 
Each use case has its own .jl file implementing the evolution models and defining the budget of resources. 
The main.jl (vaccine allocations and traffic congestion), main_social.jl (online promotion) and main_facility.jl (vaccination centers deployment) files contain all the functions needed for the BP to run (ODE solver, State-Clustering Algorithm, Column Generation, and full BP) for each associated use case. 
Finally, to run everything, each use case has a run_use_case.jl file that runs a full experiment, simply pass the argument (Time periods, total budget...) to obtain results.
