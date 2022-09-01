- message.py - TrainRequest - add current_round
- job.py - start_nodes_training_round - add current_round
- node.py - parser_task - add current_round = msg.get_param("current_round")
- job.py - line 372 - remove return

to finish job_jl and experiment_jl