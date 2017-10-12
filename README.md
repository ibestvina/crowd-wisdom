## Influence of Neighbours on One’s Own opinion <br> – Election prediction using social network topology
Election results prediction using the social network neighbourhood information is a complex network theory problem which includes advanced complex network modelling and complex network analysis techniques. Presuming that social networks display a high degree of homophily, which in binary elections leads to polarization and a split into two giant clusters, we translated this problem into a domain of cluster size inference, for these two clusters.

First, we have analysed how the social neighbourhood is related to a person's voting preference. Next, we have shown that the prediction of this preference can be done from the neighbourhood information in a straightforward manner. Finally, we have shown how, by modelling social networks with the expanded stochastic Kronecker graph model and using our novel _Them vs. us estimator_, network's vote share can successfully be estimated, even from a small and highly biased node sample. 

Author: Ivan Bestvina  
Mentor: Mile Šikić, PhD


### Contents
Supplement project code demonstrating the election results estimation from the biased network samples is presented in the [performance analysis notebook](performance_analysis.ipynb). Code here is slow, but easily readable, following the algorithms described in the thesis directly. Its purpose is to provide a deeper insight into the methods used, and to show how the performance graphs are generated from the simulations.

To execute simulations on a large scale, use the higly optimized code found in [simulations](simulations) folder:
```
python run_sims.py net_id
```
where `net_id` is the id of the network you want to simulate, specified in the [nets_to_sample.csv](simulations/nets_to_sample.csv). For a better understanding of the network and simulation parameters, consult the thesis.
