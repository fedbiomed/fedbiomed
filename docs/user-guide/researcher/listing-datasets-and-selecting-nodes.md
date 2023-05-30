# Listing Datasets and Selecting Nodes

In this article, you will learn how to list datasets that are deployed in nodes and select specific nodes to conduct your experiment. 

## Listing Datasets 

The `list()` method of the `Requests` class has been created for listing datasets on the active nodes. It sends `list` request to the nodes and waits for the reply. It gets two arguments as `nodes` and `verbose`;

* `verbose`: If it is `True`, it will print the dataset lists in table format for each node. Default is `False` 
* `nodes`: It is a list that includes the node ids to send list requests. Default is `None` and it means that it sends list requests to all activate nodes.  

It returns a python `dict` that includes datasets for each node. 

```python

from fedbiomed.researcher.requests import Requests
req = Requests()
req.list(verbose=True)

```

If you set `verbose=True` you will get the following output that shows datasets on nodes up and running. 


```
 Node: node_481d9ec3-79e5-49d1-96a2-9f4928d3ecf4 | Number of Datasets: 1 
+--------+-------------+--------+---------------+---------+
| name   | data_type   | tags   | description   | shape   |
+========+=============+========+===============+=========+
| sk     | csv         | ['sk'] | sk            | [20, 6] |
+--------+-------------+--------+---------------+---------+

2021-10-19 16:51:59,699 fedbiomed INFO - 
 Node: node_e289dfdc-4635-4c3c-938a-9548dbb85c92 | Number of Datasets: 2 
+--------+-------------+------------------------+----------------+--------------------+
| name   | data_type   | tags                   | description    | shape              |
+========+=============+========================+================+====================+
| MNIST  | default     | ['#MNIST', '#dataset'] | MNIST database | [60000, 1, 28, 28] |
+--------+-------------+------------------------+----------------+--------------------+
| sk     | csv         | ['sk']                 | sk             | [20, 6]            |
+--------+-------------+------------------------+----------------+--------------------+
```

Listing datasets technically lists active nodes in the network. When the `verbose` argument is `True` it also 
prints nodes that don't have any dataset and indicates that the node has no dataset. 

You can also list datasets in specific nodes;


```python

req.list(nodes=['node_e289dfdc-4635-4c3c-938a-9548dbb85c92'], verbose=True)

```

It will return the datasets deployed only in the node: `node_e289dfdc-4635-4c3c-938a-9548dbb85c92`

```
 Node: node_e289dfdc-4635-4c3c-938a-9548dbb85c92 | Number of Datasets: 2 
+--------+-------------+------------------------+----------------+--------------------+
| name   | data_type   | tags                   | description    | shape              |
+========+=============+========================+================+====================+
| MNIST  | default     | ['#MNIST', '#dataset'] | MNIST database | [60000, 1, 28, 28] |
+--------+-------------+------------------------+----------------+--------------------+
| sk     | csv         | ['sk']                 | sk             | [20, 6]            |
+--------+-------------+------------------------+----------------+--------------------+
```

## Selecting Nodes for the Experiment

The experiment class has `nodes` arguments to optionally select specific nodes on which the federated training will be performed. 
If you pass a non-empty list of node ids, then only the nodes that have a matching dataset and belong to the `nodes` list are selected.

Let's assume that you want to perform training only in the node  `node_e289dfdc-4635-4c3c-938a-9548dbb85c92`

```python
nodes = ['node_e289dfdc-4635-4c3c-938a-9548dbb85c92']
```

Afterwards, you need to pass the `nodes` list while you are initializing the experiment class.
The experiment will send a search request to the nodes in the `nodes` list for datasets deployed with the given tags.

```python
from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage

tags =  ['#MNIST', '#dataset']
rounds = 2

exp = Experiment(tags=tags,
                 nodes=nodes,
                 model_args=model_args,
                 training_plan_class=MyTrainingPlan,
                 training_args=training_args,
                 round_limit=rounds,
                 aggregator=FedAverage(),
                 node_selection_strategy=None)
```

The output of the initialization will be similar to the following output.

```
2021-10-19 17:06:16,599 fedbiomed INFO - Searching dataset with data tags: ['#MNIST', '#dataset'] on specified nodes: ['node_e289dfdc-4635-4c3c-938a-9548dbb85c92']
2021-10-19 17:06:16,631 fedbiomed INFO - log from: node_e289dfdc-4635-4c3c-938a-9548dbb85c92 - DEBUG Message received: {'researcher_id': 'researcher_1c4fc722-02c8-41b2-b9ed-b85d97968ba9', 'tags': ['#MNIST', '#dataset'], 'command': 'search'}
2021-10-19 17:06:26,612 fedbiomed INFO - Node selected for training -> node_e289dfdc-4635-4c3c-938a-9548dbb85c92
```
