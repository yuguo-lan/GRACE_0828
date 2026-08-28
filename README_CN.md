# GRACE - JPDC Final Experiment Code

本代码包已经清理为 JPDC 正式对比实验版本。旧 RFCSC、Random、开发期 Oort-Graph 变体、GRACE 消融 selector、utility/graph-interval 测试以及 selector microbenchmark 均已从本包删除。

## 1. 最终比较方法

| 命令名 | 论文名称 | 角色 |
|---|---|---|
| `poc` | Power-of-Choice (PoC) | loss/utility-aware baseline |
| `oort` | Oort | utility + system-aware baseline |
| `rbcs_f` | RBCS-F | fairness/resource-aware baseline |
| `mbut_cs` | MBUT-CS | structure/cluster-aware baseline |
| `divfl` | DivFL | diversity-aware baseline |
| `fedcor` | FedCor | correlation-aware baseline |
| `fedppo` | FedPPO | learning-based baseline |
| `graph_diversity` | GRACE | proposed method |

`power_of_choice` 仍作为 `poc` 的兼容别名，但正式运行建议统一使用 `poc`。

## 2. 统一入口

```bash
./run.sh help
```

正式实验入口：

```bash
./run.sh main
./run.sh noniid
./run.sh scalability
./run.sh participation
./run.sh availability
./run.sh system
./run.sh cifar100
./run.sh analyze
./run.sh stats
```

## 3. 主实验

单方法、单数据集：

```bash
./run.sh main 0 graph_diversity fashionmnist
./run.sh main 1 poc cifar10
```

全部最终方法和全部主数据集：

```bash
./run.sh main 0 all all
```

默认：MNIST/FashionMNIST/CIFAR-10，Dirichlet alpha=0.5/0.7/0.9，seeds=41/42/43，N=100，K=10。

## 4. 强 Non-IID

```bash
SEEDS="41 42 43" ALPHAS="0.3 0.5 0.7" ./run.sh noniid 0 graph_diversity fashionmnist
SEEDS="41 42 43" ALPHAS="0.3 0.5 0.7" ./run.sh noniid 1 divfl cifar10
```

用于验证严重统计异质性下的鲁棒性。

## 5. 客户端规模扩展

```bash
CLIENTS="100 200 500" SEEDS="41 42 43" ALPHA=0.3 \
./run.sh scalability 0 graph_diversity cifar10
```

用于验证客户端数量增加时的准确率、收敛、coverage/fairness 与选择开销趋势。

## 6. Participation Rate

```bash
NUM_CLIENTS=100 RATES="0.05 0.10 0.20" SEEDS="41 42 43" ALPHA=0.3 \
./run.sh participation 0 graph_diversity fashionmnist
```

严格低参与率可使用：

```bash
NUM_CLIENTS=500 RATES="0.01 0.05 0.10" SEEDS="41 42 43" ALPHA=0.3 \
./run.sh participation 0 graph_diversity fashionmnist
```

## 7. Availability / Dropout

```bash
SEEDS="41 42 43" ALPHA=0.3 \
./run.sh availability 0 graph_diversity fashionmnist
```

用于验证动态在线率和选后掉线下的鲁棒性。

## 8. System Heterogeneity

```bash
SEEDS="41 42 43" ALPHA=0.3 \
./run.sh system 0 graph_diversity fashionmnist
```

用于 homogeneous / mild / strong 系统异构条件下的对比。日志中的 system latency 是 analytical/simulated latency，不应表述成真实移动设备 wall-clock latency。

## 9. CIFAR-100 + ResNet-18

```bash
SEEDS="41 42 43" ALPHAS="0.3 0.7" \
./run.sh cifar100 0 graph_diversity
```

用于验证更复杂数据集和模型架构上的泛化能力。

## 10. 批量跑全部最终方法

```bash
METHODS="poc oort rbcs_f mbut_cs divfl fedcor fedppo graph_diversity"
for m in $METHODS; do
    SEEDS="41 42 43" ALPHA=0.3 ./run.sh scalability 0 "$m" cifar10
done
```

同样方式可以用于 `noniid`、`participation`、`availability`、`system` 和 `cifar100`。

## 11. 结果汇总与显著性

```bash
./run.sh analyze
EXP_NAME=main_multiseed ./run.sh stats
```

默认结果目录为：

```text
results_jpdc/
```

正式实验建议至少报告 mean +/- std、final/checkpoint accuracy、rounds-to-target、coverage/Jain fairness、communication volume 与 selection latency。


> Oort 说明：当前 `oort` 使用清理前原 `oort_core` 的实现；旧的另一版 `oort` 已移除。运行命令仍使用 `oort`。
