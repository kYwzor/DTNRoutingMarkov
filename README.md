# DTNRoutingMarkov
Python scripts developed for a dissertation titled **Delay Tolerant Network Routing** in the context of a Master in Informatics Engineering.

Includes modelling and analysis of a urban sensing scenario with a Delay Tolerant Network, based on a finite-state discrete-time homogeneous Markov chain model. The scripts are organized as follows:

- **markov_chain.py** - Contains the model and its corresponding analysis. 

- **communication_routing_models.py** - Contains communication models and routing strategies (used in markov_chain.py)

- **mobility_models.py** - Contains mobility models (used in markov_chain.py)

- **tpm_simulation.py** - A simulation of this model, based on its Transition Probability Matrix (used for verification)

- **entity_simulation.py** - A simulation of this model, based on the behaviour of its entities (used for verification)

All scripts were tested for Python 3.7 (check requirements.txt).
