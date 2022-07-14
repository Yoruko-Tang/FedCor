# FedCor

## Environment

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
python3 src/federated_main.py --gpu=0 --dataset=fmnist --model=mlp --mlp_layer 64 30  --epochs=500 --num_user=100 --frac=0.05 --alpha=0.2 --local_ep=3 --local_bs=64 --lr=5e-3 --schedule 150 300 --lr_decay=0.5 --optimizer=sgd --iid=0 --unequal=0  --verbose=1 --seed 1 2 3 4 5 --gpr --gpr_selection --poly_norm=0 --GPR_interval=10 --group_size=11 --warmup=15 --discount_method=time --discount=0.95
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
python3 src/federated_main.py --gpu=0 --dataset=cifar --model=cnn --kernel_sizes 3 3 3 --num_filters 32 64 64 --mlp_layer 64 --epochs=2000 --num_user=100 --frac=0.05 --local_ep=5 --local_bs=50 --lr=0.01 --lr_decay=1.0 --optimizer=sgd --reg=3e-4 --iid=0 --unequal=0 --alpha=0.2 --verbose=1 --seed 1 2 3 4 5 --gpr --gpr_selection --discount_method=time --discount=0.9 --GPR_interval=50 --group_size=51 --GPR_gamma=0.97 --poly_norm=0 --warmup=20
```

#### Pow-d

```shell
python3 src/federated_main.py --gpu=0 --dataset=cifar --model=cnn --kernel_sizes 3 3 3 --num_filters 32 64 64 --mlp_layer 64 --epochs=2000 --num_user=100 --frac=0.05 --local_ep=5 --local_bs=50 --lr=0.01 --lr_decay=1.0 --optimizer=sgd --reg=3e-4 --iid=0 --unequal=0 --alpha=0.2 --verbose=1 --seed 1 2 3 4 5 --power_d --d=10
```

#### AFL

```shell
python3 src/federated_main.py --gpu=0 --dataset=cifar --model=cnn --kernel_sizes 3 3 3 --num_filters 32 64 64 --mlp_layer 64 --epochs=2000 --num_user=100 --frac=0.05 --local_ep=5 --local_bs=50 --lr=0.01 --lr_decay=1.0 --optimizer=sgd --reg=3e-4 --iid=0 --unequal=0 --alpha=0.2 --verbose=1 --seed 1 2 3 4 5 --afl
```

