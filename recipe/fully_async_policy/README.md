# 基于verl的改造方案

## 方案

### 方案1 (StreamRL, AsyncFlow)

![StreamRL](
https://github.com/ArronHZG/verl-community/blob/recipe/async_policy/docs/StreamRL.png?raw=true)

在分离架构的基础上，修改在Rollout和Train的样本传递过程中，将离线策略生成一批global样本修改为生成一批batch的方式，实现生成和训练两阶段的高度重叠。
训练阶段一收到足够样本就开始处理，训练一定步数后，将参数同步到PS侧， Rollout在每次样本生成完成后，check是否有新的参数，如果有就进行一次同步。

### 方案2 (Mistralai, Areal)

![mistralai](
https://github.com/ArronHZG/verl-community/blob/recipe/async_policy/docs/mistralai.png?raw=true)

在分离架构的基础上，实现Rollout的partial rollout逻辑。样本仍然修改为batch的方式进行传递，实现生成和训练两阶段的高度重叠。
在参数同步方面，训练阶段主动触发Rollout的暂停，参数同步以及恢复。 Rollout使用Rollout Server的方式，支持样本生成的中断与恢复，
产生的的样本所使用的参数版本会有所不同。

### 折中

上述两种方案的核心都是将训练与生成进行overlap，核心区别主要集中在参数同步的处理方式不同，方案1需要实现PS完成参数的异步加载。
方案2使用同步的方式进行参数同步，但需要完成PartialRollout的逻辑。综合已有代码，以及社区进行中的工作，我们希望先将异步的工作流搭建完成，先以方案1进行开发，后续再进一步开发方案2。

## 设计

### 架构图

![full_async](
https://github.com/ArronHZG/verl-community/blob/recipe/async_policy/docs/full_async.svg?raw=true)

为实现纯异步训练工作流，基于已有的 one step off policy 代码，扩增实现 Rollouter 以及 Message Queue，以及对Trainer进行更新。

整体的训练流程参考StreamRL，将原有流程中生成 train_batch_size 个样本后进行下一步训练的过程，修改为流式的样本传递，train
拿到一次前向的样本后就进行样本分发（ppo_mini_batch_size*worker）。与one-step-off相比，我们将一次step的异步，继续细化到一次batch的异步。

**MessageQueue** 作为Ray的Actor存在，支持zeromq消息队列保存生成的样本，并提供给Trainer使用。Trainer 和 Rollouter 都持有
MessageQueue 的Handler，通过接口完成样本的插入与消费。

**FullyAsyncRollouter** 类似于现有的 Trainer，实现fit()工作流，循环调用 Rollout 进行样本的生成。FullyAsyncRollouter 对于已有的
vLLMAsyncRollout SGLangAsyncRollout 进行封装。

* 方案1，使用异步更新策略，FullyAsyncRollouter 根据样本生成的进展，自动访问PS，判断是否进行新的参数加载。
* 方案2，参考PR https://github.com/volcengine/verl/pull/2246 https://github.com/volcengine/verl/pull/2200 Rollout
  组件需要支持暂停及恢复，从而进行参数的更新。暂停时，需要保存进行中的rollout样本，下次继续恢复生产。

**FullyAsyncTrainer** 与当前实现类似，区别是样本的获取修改为从Queue中获取，Queue有最少batch样本就开始进行分发。rainer完成一次step的训练后，
与FullyAsyncRollouter的使用策略对应：

* 方案1，使用异步更新策略，参数产生后，主动同步到PS中。
* 方案2，直接调用Rollouter进行同步，主动通知Rollouter暂停生成，进行参数的同步更新。

## 总结

当Rollouter生产快于Trainer消费时，queue中会存在多步过期的样本，我们需要在Rollouter中设置“陈旧度 staleness”阈值，
由当前的参数版本以及生成的样本数量，决定是否要暂停生成。zeromq 的最大长度应为 staleness * total_size，并且实现基于陈旧度的拒绝策略，进行防御性编程。

* 当使用方案1时，参数的同步由FullyAsyncRollouter主动控制，触发时机取决预先设置的固定数量样本完成以及参数已就绪，产生的样本所使用的参数版本一致，
  但是避免不了长尾的问题，会有"rollout空洞"产生。

* 当使用方案2时，参数的同步会更加及时，陈旧度低的样本数量较多，但是长尾样本由不同的参数产生，长尾样本的不同token所对应的参数版本会传递给训练引擎，
  后续可以根据这一信息对loss进行加权处理。

当Rollouter生产慢于Trainer消费时，队列长时间为空，基本等价于同步训练。