## Fed-BioMed v4.1 new release

![v4.1](../assets/img/v4.1.jpg#img-centered-sm)

Fed-BioMed v4.1 is now available. Here are some key new features:

- Introducing [Scaffold `Aggregation`](https://arxiv.org/abs/1910.06378) method for PyTorch, focused to cope with the *client drift* issue, useful when dealing with heterogeneous datasets
- Adding `num_updates` as a new `training_args` Argument: `num_updates` allows you to iterate your model over a specific number of updates, regardless of the size of data across each `Node`. It is an alternative to number of epochs `epochs`
- Adding more integration tests / introducing nightly tests in order to improve code quality
- improving `Researcher` log message, by introducing `Round` number
- Bug fixes (FedProx `Aggregation` method, percentage completion logged when using Opacus, and other minor fixes)

More details about the new features can be found in the [Fed-BioMed CHANGELOG](https://github.com/fedbiomed/fedbiomed/blob/v4.1/CHANGELOG.md).
