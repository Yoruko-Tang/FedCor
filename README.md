# FedCor
The official implementation of the Paper: *FedCor: Correlation-Based Active Client Selection Strategy for Heterogeneous Federated Learning* (CVPR 22').

**Abstract:** Client-wise data heterogeneity is one of the major issues that hinder effective training in federated learning (FL). Since the data distribution on each client may vary dramatically, the client selection strategy can significantly influence the convergence rate of the FL process. Active client selection strategies are popularly proposed in recent studies. However, they neglect the loss correlations between the clients and achieve only marginal improvement compared to the uniform selection strategy. In this work, we propose FedCor---an FL framework built on a correlation-based client selection strategy, to boost the convergence rate of FL. Specifically, we first model the loss correlations between the clients with a Gaussian Process (GP). Based on the GP model, we derive a client selection strategy with a significant reduction of expected global loss in each round. Besides, we develop an efficient GP training method with a low communication overhead in the FL scenario by utilizing the covariance stationarity. Our experimental results show that compared to the state-of-the-art method, FedCor can improve the convergence rates by 34% ~ 99% and 26% ~ 51% on FMNIST and CIFAR-10, respectively.


## Required Environment

1. python 3.8
2. pytorch 1.7.0
6. cvxopt 1.2.0

## Running Experiments

Cd to the root directory of the repository and run the following command lines to start a training on Dir setting with different client selection strategies. If you want to train with 2SPC or 1SPC setting, you can replace the option ```--alpha=0.2``` with ```--shards_per_client=2``` or ```--shards_per_client=1``` respectively.

For more details about each option, see ```/src/options.py```.

### FMNIST

#### Rand

```shell
python3 src/federated_main.py --gpu=0 --dataset=fmnist --model=mlp --mlp_layer 64 30  --epochs=500 --num_user=100 --frac=0.05 --alpha=0.2 --local_ep=3 --local_bs=64 --lr=5e-3 --schedule 150 300 --lr_decay=0.5 --optimizer=sgd --iid=0 --unequal=0  --verbose=1 --seed 1 2 3 4 5
```

#### FedCor

```shell
python3 src/federated_main.py --gpu=0 --gpr_gpu=0 --dataset=fmnist --model=mlp --mlp_layer 64 30  --epochs=500 --num_user=100 --frac=0.05 --alpha=0.2 --local_ep=3 --local_bs=64 --lr=5e-3 --schedule 150 300 --lr_decay=0.5 --optimizer=sgd --iid=0 --unequal=0  --verbose=1 --seed 1 2 3 4 5 --gpr --poly_norm=0 --GPR_interval=10 --group_size=100 --GPR_gamma=0.95 --update_mean --warmup=15 --discount=0.9
```

#### Pow-d

```shell
python3 src/federated_main.py --gpu=0 --dataset=fmnist --model=mlp --mlp_layer 64 30  --epochs=500 --num_user=100 --frac=0.05 --alpha=0.2 --local_ep=3 --local_bs=64 --lr=5e-3 --schedule 150 300 --lr_decay=0.5 --optimizer=sgd --iid=0 --unequal=0  --verbose=1 --seed 1 2 3 4 5 --power_d --d=10
```

#### AFL

```shell
python3 src/federated_main.py --gpu=0 --dataset=fmnist --model=mlp --mlp_layer 64 30  --epochs=500 --num_user=100 --frac=0.05 --alpha=0.2 --local_ep=3 --local_bs=64 --lr=5e-3 --schedule 150 300 --lr_decay=0.5 --optimizer=sgd --iid=0 --unequal=0  --verbose=1 --seed 1 2 3 4 5 --afl
```



### CIFAR10

#### Rand

```shell
python3 src/federated_main.py --gpu=0 --dataset=cifar --model=cnn --kernel_sizes 3 3 3 --num_filters 32 64 64 --mlp_layer 64 --epochs=2000 --num_user=100 --frac=0.05 --local_ep=5 --local_bs=50 --lr=0.01 --lr_decay=1.0 --optimizer=sgd --reg=3e-4 --iid=0 --unequal=0 --alpha=0.2 --verbose=1 --seed 1 2 3 4 5
```

#### FedCor

```shell
python3 src/federated_main.py --gpu=0 --gpr_gpu=0 --dataset=cifar --model=cnn --kernel_sizes 3 3 3 --num_filters 32 64 64 --mlp_layer 64 --epochs=2000 --num_user=100 --frac=0.05 --local_ep=5 --local_bs=50 --lr=0.01 --lr_decay=1.0 --optimizer=sgd --reg=3e-4 --iid=0 --unequal=0 --alpha=0.2 --verbose=1 --seed 1 2 3 4 5 --gpr --discount=0.9 --GPR_interval=50 --group_size=500 --GPR_gamma=0.99 --poly_norm=0 --update_mean --warmup=20
```

#### Pow-d

```shell
python3 src/federated_main.py --gpu=0 --dataset=cifar --model=cnn --kernel_sizes 3 3 3 --num_filters 32 64 64 --mlp_layer 64 --epochs=2000 --num_user=100 --frac=0.05 --local_ep=5 --local_bs=50 --lr=0.01 --lr_decay=1.0 --optimizer=sgd --reg=3e-4 --iid=0 --unequal=0 --alpha=0.2 --verbose=1 --seed 1 2 3 4 5 --power_d --d=10
```

#### AFL

```shell
python3 src/federated_main.py --gpu=0 --dataset=cifar --model=cnn --kernel_sizes 3 3 3 --num_filters 32 64 64 --mlp_layer 64 --epochs=2000 --num_user=100 --frac=0.05 --local_ep=5 --local_bs=50 --lr=0.01 --lr_decay=1.0 --optimizer=sgd --reg=3e-4 --iid=0 --unequal=0 --alpha=0.2 --verbose=1 --seed 1 2 3 4 5 --afl
```

### Update
#### 02/10/2023
1. Refine codes and improve the efficiency and stability when training and utilizing GP.
2. Support GP on GPU.
3. Enable updates of the mean in GP.