# Introduction to Theoretical Investigations

This repo is intended to serve as a means of communication between my Azure server and my local PC. The idea is to commit data generation form server to this repo, and then to my local machine.

In the future, I shall include a better description about the contents of the data, for now, the idea is to mediate.

## List of files

* ```IsingModel.py```: This file is intended to have the core routines for MCMC simulation of a Binary Alloy using a BEG Model

* ```IsingSolver.py```: This file is intended to have all routines related to data generation: from magnetisation profile, to interpolation data, and plotting routines

* ```InitServer.sh```: This shell script is intended to set up the development environment in my server: from installing anaconda and Twilio, to syncing to this repo using git.

* ```GenerateData.sh```: This shell script should execute the codes for data generation, and commit the csv files with data to this repo.
