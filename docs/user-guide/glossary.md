# Glossary

Here below the glossary used for Fed-BioMed :


* **experiment** : orchestrates the rounds during the federated learning, on the available nodes
    -	it includes : training plan, model, federated trainer, training parameters, model parameters, set of input data, results
    -	an experiment is unique (cannot be replayed) and is over when converged
    -	status : running and then done


* **training** : as commonly used in ML, process of feeding a model with data to improve its accuracy on some task.
* **validation** : process of giving a heuristic information on the accuracy of a model during training.
* **testing** : process of assessing the accuracy of a model after training, on holdout samples different from the one that were used for training. Not implemented yet in Fed-BioMed.


* **job** : not a researcher notion. Interface between the researcher and the nodes of an experiment. It triggers the local work for all sampled nodes at each round.
* **round** : everything included in choice of the nodes, perform local work on the nodes, sending back whatever information is required, server performs the aggregation
    - current Round() class on node corresponds to local work
* **parameter update** : an update of the ML model parameters during the training loop, which usually corresponds to the processing of one batch of data
* **epoch** : a number of parameter updates equivalent to processing the entire dataset exactly once


* **researcher** (technical) : entity that defines and executes an experiment
* **node** (technical) : entity with tagged datasets that replies to researcher queries and performs local work