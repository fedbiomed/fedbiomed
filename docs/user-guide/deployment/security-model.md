# Fed-BioMed security model

This page gives an overview of Fed-BioMed security model. A more complete and formal description of the security model underlying Fed-BioMed is currently work in progress.

## Summary

!!! info "Fed-BioMed empowers the node sites"
    In a Fed-BioMed instance, nodes have control. There is no notion of trusted party that has full control or full access to the other parties. The researcher can only train a model authorized by the node, on data explicitely shared by the node.

!!! info "Fed-BioMed minimizes firewall filters"
    Fed-BioMed nodes and researcher only need one outbound VPN port to one server for running

!!! info "Fed-BioMed offers high protection against outsiders"
    Fed-BioMed offers high protection against attacks coming from outside of the Fed-BioMed instance by isolating all communications between the components inside a VPN.

!!! info "Fed-BioMed protects from major attacks by insiders"
    Fed-BioMed also identifies possible attacks coming from one of the Fed-BioMed instance components and already offers protection against major attack scenarios.


## Assets, threats, vulnerabilities

Fed-BioMed security assets (as defined in [ENISA glossary](https://www.enisa.europa.eu/topics/risk-management/current-risk/risk-management-inventory/glossary)) and their assessed value are:

* **node data:** the primary goal of Fed-BioMed is to protect the data of the participating nodes 
* **host machines:** they are an indirect asset (*infrastructure asset*) of Fed-BioMed as they host the other assets. Thus, the protection of the machines hosting the node, researcher and network components is at least as important as protecting the assets they host. Moreover, Fed-BioMed software should not be a vector to compromise the host machine or other machines/assets on the host site.
* **experiment inputs** (training plan source code, optional custom strategy/optimizer source code, training parameters and model hyper-parameters) **and outputs** (final trained model, experiment intermediate results, local training updates from nodes): the final trained model parameters are an important asset as they are the main output of the software. Experiment inputs and intermediate results are necessary to compute the final trained model, and the users may value intellectual property on them.

Fed-BioMed identified threats are:

* **outsiders:** they include all the machines/people that do not belong to the Fed-BioMed instance (all but the Fed-BioMed components). They are considered to be the most likely adversaries, conducting active attacks (malicious). They mostly try to breach confidentiality of data, but may attempt any type of impact on assets.
* **insiders:** they are the members of the Fed-BioMed instance (node, researcher, network). They are considered to be less likely adversaries. Our current security model addresses in priority the case of honest but curious nodes and network (parties do not attempt at modifying the protocol for attacks), while the researcher may be malicious. Attacks are primarily aimed at breach data confidentiality, although they may also attempt to other kind of assets.

Fed-BioMed main identified vulnerabilities are:

* `1.` federated learning: honest but curious researcher can attempt privacy inference attacks on local model parameters sent by the nodes and try to gain some knowledge about the nodes' data
* `2.` infrastructure: honest but curious node or network man in the middle (MITM) can listen to MQTT/restful exchanges between parties or access them directly on MQTT/restful. It then learns queries and results of the trainings performed by the nodes. The primary interest is to learn the local model parameters from other nodes and attempt attacks mentioned in `1.`. Network can also learn global model parameters and attempt attacks mentioned in `8.`
* `3.` federated learning: malicious researcher authorized to train on a node can send malicious training plan code to breach the assets. Typically, it tries to leak data from the nodes.
* `4.` infrastructure: malicious insider man in the middle (MITM) can spoof the researcher or the network to execute training commands on the node (possibly `3.`)

Fed-BioMed other vulnerabilities include:

* `5.` federated learning: advanced attacks such as model poisoning, free-riding attacks, etc.
* `6.` infrastructure: outsider may attempt penetration attacks on a VPN endpoint
* `7.` infrastructure: insider may attempt penetration attacks on another component of the Fed-BioMed instance
* `8.` federated learning: honest but curious nodes can attempt privacy inference attacks on global model parameters sent by the researcher and try to gain some knowledge about the other nodes' data
* `9.` inference attacks on the final trained model: a malicious outsider that duly receives a copy of the final trained model for using it may try attacks from `1.`. This case is considered out of scope of this analysis, as it occurs outside of Fed-BioMed. Same precautions should be taken as for any machine learning model.

## Addressing the vulnerabilities

Fed-BioMed addresses the above vulnerabilities in the following way:

* **secure aggregation** and **differential privacy** offer options to reduce the risk coming from `1.`
* exploiting `2.` or `4.` would enable a node or the network component to execute same commands (training) or retrieve same information (local training updates from node) as the researcher, but no more. This is why implementation of secure communication inside the VPN was not prioritized by Fed-BioMed. Nevertheless, it is in the midterm roadmap.
* **model approval functionality** addresses `3.` by enabling each node site to review and authorize a training plan before it can train on the node.
* advanced federated learning attacks from `5.` will be further addressed in future releases with innovative functions. Stay tuned.
* Fed-BioMed seeks to offer minimal attack surface to penetration attacks from `6.`: the only network communication between the components is through the **WireGuard VPN**. Moreover, node sites that hold the data only have outbound connections to further reduce the attack surface on nodes.
* Fed-BioMed seeks to reduce the attack surface to penetration attacks from `7.` by design: nodes and researcher only have outbound communications to the MQTT/restful (except the temporary, authenticated and specialized MP-SPDZ inbound and outboung connections for negotiating secure aggregation keys). Also, the node only accepts a limited set of commands from a legitimate researcher. There is no notion of trusted party that have full control over the nodes or the node data in a Fed-BioMed instance. Finally, our future implementation of secure communication inside the VPN will further reduce this risk.
* attacks on global updates from `8.` are considered more complicated than attacks on local updates. Currently, implemented Local and Central Differential Privacy are valid mechanisms to protect against these attacks, and other specific defense strategies will be further addressed in future releases. 
