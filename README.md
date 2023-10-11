[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Curriculum Learning-Inspired Efficient Client Sampling with Reinforcement Agent in Federated Learning

This archive is distributed in association with the [INFORMS Journal on Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

This repository contains supporting material for the paper *Curriculum Learning-Inspired Efficient Client Sampling with Reinforcement Agent in Federated Learning* by Quanquan Hu, Xuefeng Liu, Shaojie Tang, Jianwei Niu and Zhangmin Huang.

The software and data in this repository are a snapshot of the software and data that were used in the research reported on in the paper.

## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2022.0150

https://doi.org/10.1287/ijoc.2022.0150.cd

Below is the BibTex for citing this snapshot of the respoitory.

```
@misc{fedecs-ijoc-repo-22,
  author =        {Quanquan Hu and Xuefeng Liu and Shaojie Tang and Jianwei Niu and Zhangmin Huang},
  publisher =     {INFORMS Journal on Computing},
  title =         {{Curriculum Learning-Inspired Efficient Client Sampling with Reinforcement Agent in Federated Learning}},
  year =          {2022},
  doi =           {10.1287/ijoc.2022.0150.cd},
  url =           {https://github.com/INFORMSJoC/2022.0150}
}  
```

## Description

This repository contains the source code of FedECS proposed in the paper *Curriculum Learning-Inspired Efficient Client Sampling with Reinforcement Agent in Federated Learning*.

## Content

The [src](src) folder contains Python implementation of the models discussed in the paper.

- [federated_main.py](src/federated_main.py): The main function of the federated training process.
- [client.py](src/client.py): The client class file that includes how the local training executes.
- [models.py](src/models.py): The deep models used in the paper.
- [sampling.py](src/sampling.py): The client data distribution construction tools.
- [options.py](src/options.py): The hyper-parameters used by the proposed approaches.
- [utils.py](src/utils.py): The dataset reading tools.
- [ddqn.py](src/ddqn.py): The DRL agent model.

The [data](data) folder contains the datasets used in the paper.

## Requirements

Running the codes requires Python 3.6 and some packages. When Python 3.6 is installed, execute the following command.

```shell
pip3 install -r requirements.txt
```

## Replicating

The results of expriments on the dataset CIFAR-100 and the deep model WRN-28-10 can be obtained by running the following commands.

```shell
cd ./src
```

##### cifar100 on WRN:

```shell
python federated_main.py --model=WRN --dataset=cifar100 --method=fedecs --interacts=5000 --wd=5e-4 --local_ep=8 --local_bs=50 --num_users=250 --frac=0.04
python federated_main.py --model=WRN --dataset=cifar100 --method=fedavg --interacts=5000 --wd=5e-4 --local_ep=8 --local_bs=50 --num_users=250 --frac=0.04
```

After the experiments are completed, the result graphs can be get  by running the following commands.

```shell
cd ./src
python plt.py
```

## Support

For support in using this software, submit an issue.
