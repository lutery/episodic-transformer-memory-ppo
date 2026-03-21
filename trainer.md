# 讲解一下选中的代码
这三行代码的作用是：

**提前构造好“每个时间步应该取哪一段 memory window”的索引表。**

位置在 trainer.py。

也就是说，后面当第 $t$ 步要从整段 episode memory 里取出最近 `memory_length` 个位置时，不用每次现算，直接查这张表就行。

---

# 原代码

在 trainer.py：

```python
repetitions = torch.repeat_interleave(torch.arange(0, self.memory_length).unsqueeze(0), self.memory_length - 1, dim = 0).long()
self.memory_indices = torch.stack([torch.arange(i, i + self.memory_length) for i in range(self.max_episode_length - self.memory_length + 1)]).long()
self.memory_indices = torch.cat((repetitions, self.memory_indices))
```

---

# 先说最终目的

假设：

- `memory_length = 4`
- `max_episode_length = 7`

最终得到的是：

```python
[
 [0, 1, 2, 3],
 [0, 1, 2, 3],
 [0, 1, 2, 3],
 [0, 1, 2, 3],
 [1, 2, 3, 4],
 [2, 3, 4, 5],
 [3, 4, 5, 6]
]
```

这正是注释里写的结果，见 trainer.py。

这个矩阵的含义是：

- 第 0 步：看 `[0,1,2,3]`
- 第 1 步：看 `[0,1,2,3]`
- 第 2 步：看 `[0,1,2,3]`
- 第 3 步：看 `[0,1,2,3]`
- 第 4 步：看 `[1,2,3,4]`
- 第 5 步：看 `[2,3,4,5]`
- 第 6 步：看 `[3,4,5,6]`

本质上是一个**滑动窗口索引表**。

---

# 为什么前几步都一样？

因为一开始 episode 还没走够 `memory_length` 步，窗口左边不够长。  
所以代码采取的策略是：

**前几步都固定取前 `memory_length` 个槽位，再用 `memory_mask` 屏蔽掉那些实际上还不存在的历史。**

也就是说：

- 索引先统一拿 `[0,1,2,3]`
- 但真正哪些位置可见，由 `memory_mask` 控制

这就是“**索引补齐 + mask 屏蔽**”的做法。

---

# 逐行拆开讲

---

## 第 1 行

在 trainer.py：

```python
repetitions = torch.repeat_interleave(torch.arange(0, self.memory_length).unsqueeze(0), self.memory_length - 1, dim = 0).long()
```

---

### 第一步：`torch.arange(0, self.memory_length)`

如果：

```python
self.memory_length = 4
```

那么：

```python
torch.arange(0, 4)
```

得到：

```python
[0, 1, 2, 3]
```

---

### 第二步：`.unsqueeze(0)`

把它从一维变成二维：

```python
[[0, 1, 2, 3]]
```

shape 从：

$$
(4,)
$$

变成：

$$
(1,4)
$$

---

### 第三步：`repeat_interleave(..., self.memory_length - 1, dim=0)`

这里 `self.memory_length - 1 = 3`。

于是把这一行重复 3 次：

```python
[
 [0, 1, 2, 3],
 [0, 1, 2, 3],
 [0, 1, 2, 3]
]
```

这部分就是前面说的“前几步固定窗口”。

---

## 第 2 行

在 trainer.py：

```python
self.memory_indices = torch.stack([torch.arange(i, i + self.memory_length) for i in range(self.max_episode_length - self.memory_length + 1)]).long()
```

这行是在构造真正的**滑动窗口**部分。

继续假设：

- `memory_length = 4`
- `max_episode_length = 7`

那么：

```python
range(self.max_episode_length - self.memory_length + 1)
=
range(7 - 4 + 1)
=
range(4)
```

也就是：

```python
i = 0, 1, 2, 3
```

---

### 对每个 `i` 生成一个窗口

#### 当 `i = 0`
```python
torch.arange(0, 4) -> [0, 1, 2, 3]
```

#### 当 `i = 1`
```python
torch.arange(1, 5) -> [1, 2, 3, 4]
```

#### 当 `i = 2`
```python
torch.arange(2, 6) -> [2, 3, 4, 5]
```

#### 当 `i = 3`
```python
torch.arange(3, 7) -> [3, 4, 5, 6]
```

---

### `torch.stack(...)` 之后

得到：

```python
[
 [0, 1, 2, 3],
 [1, 2, 3, 4],
 [2, 3, 4, 5],
 [3, 4, 5, 6]
]
```

这就是完整的“随着时间向右滑动”的窗口。

---

## 第 3 行

在 trainer.py：

```python
self.memory_indices = torch.cat((repetitions, self.memory_indices))
```

把前两部分拼接起来：

### `repetitions`
```python
[
 [0, 1, 2, 3],
 [0, 1, 2, 3],
 [0, 1, 2, 3]
]
```

### `self.memory_indices`
```python
[
 [0, 1, 2, 3],
 [1, 2, 3, 4],
 [2, 3, 4, 5],
 [3, 4, 5, 6]
]
```

拼起来后：

```python
[
 [0, 1, 2, 3],
 [0, 1, 2, 3],
 [0, 1, 2, 3],
 [0, 1, 2, 3],
 [1, 2, 3, 4],
 [2, 3, 4, 5],
 [3, 4, 5, 6]
]
```

shape 是：

$$
(max\_episode\_length,\ memory\_length)
$$

在这个例子中就是：

$$
(7,4)
$$

---

# 这张表后面怎么用

在 trainer.py：

```python
self.buffer.memory_mask[:, t] = self.memory_mask[torch.clip(self.worker_current_episode_step, 0, self.memory_length - 1)]
self.buffer.memory_indices[:, t] = self.memory_indices[self.worker_current_episode_step]
```

意思是：

- 当前 worker 正处于 episode 的第几步
- 就去 `self.memory_indices` 里取对应那一行
- 那一行就告诉你：应该从整段 episodic memory 中切哪几个位置

然后在 trainer.py：

```python
sliced_memory = batched_index_select(self.memory, 1, self.buffer.memory_indices[:,t])
```

真正根据索引把 memory window 切出来。

---

# 这段设计的核心思想

## 1. `self.memory`
存的是整段 episode 的 memory：

shape 大致是：

$$
(num\_workers,\ max\_episode\_length,\ num\_blocks,\ embed\_dim)
$$

## 2. 但 transformer 每次不看整段
它只看最近 `memory_length` 长度的窗口。

## 3. 所以需要一张“窗口索引表”
这三行就是在预先生成这张表。

---

# 为什么不直接每次现算

当然可以在每个 step 里动态算，比如：

```python
start = max(0, t - memory_length + 1)
...
```

但这里预先算好有几个好处：

- 更直观
- 后面直接索引即可
- 逻辑统一
- 配合 `memory_mask` 更方便

---

# 一个更直观的类比

假设一本书有 7 页，你每次最多只能看连续 4 页：

- 第 0 秒：看 0-3 页
- 第 1 秒：看 0-3 页
- 第 2 秒：看 0-3 页
- 第 3 秒：看 0-3 页
- 第 4 秒：看 1-4 页
- 第 5 秒：看 2-5 页
- 第 6 秒：看 3-6 页

这三行代码就是在提前做这样一张“看哪几页”的表。

---

# 一句话总结

这三行代码就是：

**为 episode 中每一个时间步，预生成一个长度为 `memory_length` 的滑动窗口索引，用来从完整 episodic memory 中切出当前要送给 Transformer 的那段历史。**

如果你愿意，我下一条可以继续直接结合一个具体 worker 的第 0、1、2、4 步，讲 `memory_indices + memory_mask` 是如何一起工作的。


# batched_index_select(samples["memories"], 1, samples["memory_indices"])
先看你选中的这句，在 trainer.py：

```python
memory = batched_index_select(samples["memories"], 1, samples["memory_indices"])
```

它的作用一句话概括就是：

**从每条样本所属的整段 episode memory 里，按该样本自己的窗口索引，切出当前这条样本需要送进 Transformer 的那一小段历史记忆。**

---

# 一、先解释这句代码在干什么

## 1. `samples["memories"]` 是什么
它不是当前时刻的一小段窗口，而是：

**这条样本所属 episode 的完整 memory 张量。**

shape 大致是：

$$
(B,\ \text{max\_episode\_length},\ \text{num\_blocks},\ \text{embed\_dim})
$$

其中 $B$ 是当前 mini-batch 大小。

也就是说，每条样本先拿到了“整集的记忆库”。

---

## 2. `samples["memory_indices"]` 是什么
它是每条样本应该取哪几个时间位置的索引窗口。

shape 大致是：

$$
(B,\ \text{memory\_length})
$$

例如某条样本可能是：

```python
[12, 13, 14, 15]
```

表示这条样本在训练时要查看 episode memory 中第 12 到 15 号位置。

---

## 3. `batched_index_select(..., 1, ...)` 做了什么
这里是在第 `1` 维做 batched gather。

- 第 0 维：batch 维
- 第 1 维：episode 时间维
- 后面两维：`num_blocks` 和 `embed_dim`

所以它的意思是：

> 对 batch 里的每一条样本，都用它自己的 `memory_indices`，从自己的完整 episode memory 中取出对应的时间窗口。

最终得到的 `memory` shape 大致是：

$$
(B,\ \text{memory\_length},\ \text{num\_blocks},\ \text{embed\_dim})
$$

这个结果才是后面真正喂给模型的记忆窗口，见 trainer.py。

---

# 二、为什么这里还要再切一次 memory

因为训练阶段的 mini-batch 已经被打乱了。

也就是说，当前 mini-batch 里的每条样本：

- 来自不同 worker
- 可能来自不同 episode
- 也可能来自 episode 的不同时间步

所以不能简单地拿一个统一窗口给所有样本。  
必须对**每一条样本单独按自己的索引去切自己的记忆**。

这就是这句代码的核心意义。

---

# 三、`samples` 这个字典从哪来

在 trainer.py：

```python
mini_batch_generator = self.buffer.mini_batch_generator()
for mini_batch in mini_batch_generator:
```

这里的 `samples` 实际就是 `mini_batch`。

它是在 buffer.py 的 `prepare_batch_dict()` 里先整理，  
再在 buffer.py 的 `mini_batch_generator()` 里按 mini-batch 切出来的。

---

# 四、`samples` 中每个成员的定义、来源、作用

下面按 `_train_mini_batch()` 实际会用到的项来讲。

---

## 1. `samples["obs"]`

### 来源
来自 buffer.py 的：

```python
"obs": self.obs,
```

而 `self.obs` 是在采样阶段写进去的，见 trainer.py：

```python
self.buffer.obs[:, t] = torch.tensor(self.obs)
```

### 含义
当前样本对应时刻的观测。

### shape
- 向量环境时：  
  $$ (B,\ obs\_dim) $$
- 图像环境时：  
  $$ (B,\ C,\ H,\ W) $$

### 作用
作为模型当前时刻的输入，送到 [model.py](model.py#L78) 的 `forward()`。

---

## 2. `samples["actions"]`

### 来源
来自 [buffer.py](buffer.py#L51) 的：

```python
"actions": self.actions,
```

采样时写入，见 [trainer.py](trainer.py#L215)：

```python
self.buffer.actions[:, t] = torch.stack(actions, dim=1)
```

### 含义
这条样本在采样时，旧策略实际执行过的动作。

### shape
$$
(B,\ \text{action\_branches})
$$

当前项目多数是单离散动作，所以常见是：

$$
(B,\ 1)
$$

### 作用
训练时用于重新计算新策略下该动作的 `log_prob`，见 [trainer.py](trainer.py#L302-L306)。

---

## 3. `samples["values"]`

### 来源
来自 [buffer.py](buffer.py#L52) 的：

```python
"values": self.values,
```

采样时写入，见 [trainer.py](trainer.py#L217)：

```python
self.buffer.values[:, t] = value
```

### 含义
旧策略在采样时，对该状态给出的 value 估计。

### shape
$$
(B,)
$$

### 作用
用于 PPO 的 value clipping 和构造 return，见 [trainer.py](trainer.py#L318-L321)。

---

## 4. `samples["log_probs"]`

### 来源
来自 [buffer.py](buffer.py#L53) 的：

```python
"log_probs": self.log_probs,
```

采样时写入，见 [trainer.py](trainer.py#L216)：

```python
self.buffer.log_probs[:, t] = torch.stack(log_probs, dim=1)
```

### 含义
旧策略在采样时，对当时动作的对数概率。

### shape
$$
(B,\ \text{action\_branches})
$$

### 作用
训练时和新策略的 `log_probs` 做差，得到：

```python
log_ratio = log_probs - samples["log_probs"]
ratio = torch.exp(log_ratio)
```

见 [trainer.py](trainer.py#L311-L312)。

这是 PPO 的核心比率：

$$
\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{old}}(a|s)}
$$

---

## 5. `samples["advantages"]`

### 来源
来自 [buffer.py](buffer.py#L54) 的：

```python
"advantages": self.advantages,
```

由 GAE 算出，见 [buffer.py](buffer.py#L95-L109)，调用在 [trainer.py](trainer.py#L249)。

### 含义
每条样本的 advantage 值。

### shape
$$
(B,)
$$

### 作用
用于 PPO policy loss，见 [trainer.py](trainer.py#L309-L316)。

它衡量：

> 这个动作相对 baseline 来说，是比预期更好，还是更差。

---

## 6. `samples["memory_mask"]`

### 来源
来自 [buffer.py](buffer.py#L56) 的：

```python
"memory_mask": self.memory_mask,
```

采样时写入，见 [trainer.py](trainer.py#L198)：

```python
self.buffer.memory_mask[:, t] = self.memory_mask[torch.clip(self.worker_current_episode_step, 0, self.memory_length - 1)]
```

### 含义
当前这条样本的记忆窗口里，哪些位置是真实历史，哪些只是 padding/占位。

### shape
$$
(B,\ \text{memory\_length})
$$

### 作用
在 attention 里屏蔽无效位置，见 [transformer.py](transformer.py#L58-L60)。

也就是告诉模型：

- 哪些历史可以看
- 哪些位置不能看

---

## 7. `samples["memory_index"]`

### 来源
来自 [buffer.py](buffer.py#L57) 的：

```python
"memory_index": self.memory_index,
```

它在 `mini_batch_generator()` 里不会直接放进最终 `mini_batch`，而是用来查表，见 [buffer.py](buffer.py#L86-L89)。

### 含义
当前样本属于 `self.memories` 列表中的哪一个完整 episode memory。

### shape
$$
(B,)
$$

### 作用
它是一个“episode memory 指针”。

作用不是直接喂模型，而是为了找到这条样本对应的整段 `memories`。

---

## 8. `samples["memories"]`

### 来源
它不是 `prepare_batch_dict()` 直接存进去的，而是在 [buffer.py](buffer.py#L86-L89) 动态生成：

```python
if key == "memory_index":
    mini_batch["memories"] = self.memories[value[mini_batch_indices]]
```

### 含义
当前 mini-batch 每条样本所属 episode 的完整 memory。

### shape
$$
(B,\ \text{max\_episode\_length},\ \text{num\_blocks},\ \text{embed\_dim})
$$

### 作用
作为“完整记忆库”，再配合 `samples["memory_indices"]` 切出真正的窗口。  
也就是你选中的那句代码的输入之一。

---

## 9. `samples["memory_indices"]`

### 来源
来自 [buffer.py](buffer.py#L58) 的：

```python
"memory_indices": self.memory_indices,
```

采样时写入，见 [trainer.py](trainer.py#L199)：

```python
self.buffer.memory_indices[:, t] = self.memory_indices[self.worker_current_episode_step]
```

### 含义
当前样本该查看 episode memory 的哪些时间位置。

### shape
$$
(B,\ \text{memory\_length})
$$

### 作用
配合 `samples["memories"]` 截取窗口：

```python
memory = batched_index_select(samples["memories"], 1, samples["memory_indices"])
```

---

# 五、把 `samples` 的整体流程串起来

你可以把它理解成下面这条链路：

## 第一步：采样阶段
在 trainer.py 中，环境 rollout 时把这些东西记进 buffer：

- 当前观测 `obs`
- 当时动作 `actions`
- 当时旧策略的 `log_probs`
- 当时旧 value `values`
- 当前样本应使用的 `memory_mask`
- 当前样本应使用的 `memory_indices`
- 当前样本属于哪个完整 episode memory：`memory_index`

---

## 第二步：buffer 展平
在 buffer.py 中，把 `(n_workers, worker_steps, ...)` 展平成 `(batch_size, ...)`。

---

## 第三步：mini-batch 随机采样
在 buffer.py 中：

- 随机打乱样本
- 根据 `memory_index` 找到每条样本所属的完整 episode memory
- 生成 `samples["memories"]`

---

## 第四步：训练时再切窗口
在 trainer.py：

```python
memory = batched_index_select(samples["memories"], 1, samples["memory_indices"])
```

把每条样本真正需要的那段历史切出来。

---

## 第五步：喂给模型
在 trainer.py：

```python
policy, value, _ = self.model(samples["obs"], memory, samples["memory_mask"], samples["memory_indices"])
```

---

# 六、最关键的一点：为什么设计成 `memories + memory_indices`

因为当前项目不是 RNN 那种“每一步只传一个 hidden state”，而是 Transformer memory。

Transformer 需要的是：

- 一段历史窗口
- 而不是单个上一时刻 hidden state

所以这里必须保存：

1. **完整 episode memory**
2. **当前样本该取哪段窗口**

这就是 `samples["memories"]` 和 `samples["memory_indices"]` 同时存在的原因。

---

# 七、一个超直观小例子

假设当前 mini-batch 里某条样本：

- 来自第 2 个 episode
- 当前时刻是第 5 步
- `memory_length = 4`

那么：

## `samples["memories"][k]`
可能是这整个 episode 的完整 memory：

```python
shape = (max_episode_length, num_blocks, embed_dim)
```

## `samples["memory_indices"][k]`
可能是：

```python
[2, 3, 4, 5]
```

## 选中的代码做完后
就得到：

```python
memory[k] = samples["memories"][k, [2,3,4,5], :, :]
```

这就是该样本真正要喂给 Transformer 的历史窗口。

---

# 八、总结版表格

| 成员 | 来源 | 含义 | 作用 |
|---|---|---|---|
| `obs` | rollout 采样 | 当前观测 | 模型输入 |
| `actions` | rollout 采样 | 旧策略执行动作 | 计算新 `log_prob` |
| `values` | rollout 采样 | 旧 value 估计 | value loss / clipping |
| `log_probs` | rollout 采样 | 旧策略动作对数概率 | PPO ratio |
| `advantages` | GAE | 动作优势 | policy loss |
| `memory_mask` | rollout 时按步生成 | 哪些历史位置有效 | attention mask |
| `memory_index` | rollout 时记录 | 样本属于哪个 episode memory | 查找完整 memory |
| `memories` | mini-batch 生成时查出来 | 样本所属完整 episode memory | 再切出窗口 |
| `memory_indices` | rollout 时记录 | 当前样本该取哪段记忆 | 从完整 memory 截窗口 |

---

# 九、一句话总结你选中的代码

在 trainer.py 这句代码的本质就是：

**根据每条样本自己的 `memory_indices`，从它所属 episode 的完整 `memories` 中截出一个长度为 `memory_length` 的历史窗口，供 Transformer 使用。**

如果你愿意，我下一条可以继续直接拿一个具体 shape 例子，比如：

- `B=3`
- `memory_length=4`
- `num_blocks=2`
- `embed_dim=8`

把这句 `batched_index_select()` 的输入输出张量形状完整画出来。

# 看我选中的代码，我标识的注释中的shape是否正确？如果不正确为什么不正确？
你选中的这段注释里，`shape` 有一部分是**不正确的**，主要问题在于你把 `log_probs` 和 `entropies` 在“列表阶段”的维度理解成了最终张量维度，而且把维度顺序写反了。

我直接按代码逐行校正。

---

## 先给结论

你这几句里：

```python
# log_probs： 存储每一个动作的动作概率的对数值，shape 应该是（num_actions_branchs, B）
# entropies： 存储每一个动作分支的熵值，shape 应该是（num_actions_branchs, B）
```

**这两句不准确。**

更准确地说：

1. 在 `append` 之前，`log_probs` 和 `entropies` 是 **Python list**，不是张量，严格说“shape”这个词还不适用
2. 列表中的**每个元素** shape 是：
   $$
   (B,)
   $$
3. `torch.stack(..., dim=1)` 之后，才会变成：
   $$
   (B,\ num\_action\_branches)
   $$
   不是你写的：
   $$
   (num\_action\_branches,\ B)
   $$

---

# 逐行看

代码在 trainer.py。

---

## 1. `policy` 是什么

前面一行：

```python
policy, value, _ = self.model(...)
```

在 model.py 中，`policy` 返回的是：

```python
pi = [Categorical(logits=branch(h_policy)) for branch in self.policy_branches]
```

所以 `policy` 是一个 **list**，里面每个元素都是一个 `Categorical` 分布对象。

如果当前动作空间只有一个离散动作头，那么：

- `len(policy) = 1`

如果是 multi-discrete 动作空间，那么：

- `len(policy) = num_action_branches`

---

## 2. `for i, policy_branch in enumerate(policy):`

这一层循环是在遍历**每个动作分支**。

所以这里的 `i` 不是 batch 维，而是：

- 第 0 个动作分支
- 第 1 个动作分支
- 第 2 个动作分支
- ...

---

## 3. `policy_branch.log_prob(samples["actions"][:, i])` 的 shape

对一个 `Categorical` 分布来说：

```python
policy_branch.log_prob(actions)
```

如果 `actions` 的 shape 是：

$$
(B,)
$$

那么返回的 `log_prob` shape 也是：

$$
(B,)
$$

因为它是：

**batch 中每个样本，在该动作分支上，对应动作的对数概率。**

所以这里每次 `append` 进去的单个元素 shape 是：

$$
(B,)
$$

---

## 4. `policy_branch.entropy()` 的 shape

同理，`Categorical.entropy()` 返回的是：

**batch 中每个样本该动作分布的熵**

shape 也是：

$$
(B,)
$$

所以这里每次 append 进去的单个元素也是一维向量：

$$
(B,)
$$

---

# 关键点：列表阶段没有“整体 shape”

在这两行之后：

```python
log_probs.append(...)
entropies.append(...)
```

此时：

- `log_probs` 是 Python list
- `entropies` 也是 Python list

如果有 3 个动作分支，那它们看起来更像：

```python
log_probs = [
    tensor(shape=(B,)),
    tensor(shape=(B,)),
    tensor(shape=(B,))
]
```

所以更准确的说法应该是：

- `log_probs` 是长度为 `num_action_branches` 的列表
- 列表中每个元素 shape 是 `(B,)`

而不是直接说它的 shape 是 `(num_action_branches, B)`。

---

# 5. `torch.stack(log_probs, dim=1)` 之后的 shape

这一句很关键：

```python
log_probs = torch.stack(log_probs, dim=1)
```

假设有：

- `num_action_branches = A`
- 每个元素 shape 是 `(B,)`

如果你 stack 在 `dim=1`，结果是：

$$
(B,\ A)
$$

不是：

$$
(A,\ B)
$$

---

## 为什么是 `(B, A)` 不是 `(A, B)`

因为原来每个元素是 `(B,)`，相当于一维张量。

把多个 `(B,)` 的张量在 `dim=1` 上堆叠，插入的是第 2 个维度，所以结果是：

```python
(B, A)
```

举个例子：

```python
t1.shape = (5,)
t2.shape = (5,)
t3.shape = (5,)
torch.stack([t1, t2, t3], dim=1).shape == (5, 3)
```

所以你的注释中：

```python
shape 应该是（num_actions_branchs, B）
```

是错的。

正确是：

```python
(B, num_action_branches)
```

---

# 6. `entropies = torch.stack(entropies, dim=1).sum(1).reshape(-1)`

这一句可以拆开看。

---

## 第一步：`torch.stack(entropies, dim=1)`

和 `log_probs` 一样，stack 后 shape 是：

$$
(B,\ A)
$$

其中：
- `B` = batch size
- `A` = 动作分支数

---

## 第二步：`.sum(1)`

沿着动作分支维度求和，也就是把每个样本所有动作分支的熵加起来：

$$
(B,\ A) \to (B,)
$$

---

## 第三步：`.reshape(-1)`

这里只是确保它是一维向量：

$$
(B,)
$$

所以最终：

```python
entropies.shape == (B,)
```

---

# 7. 为什么这里要 `sum(1)`

因为如果动作空间是 multi-discrete，那么总策略其实是多个独立分支分布组成的。

这时一个样本的总 entropy，通常取各分支 entropy 的和：

$$
H(\pi) = \sum_i H(\pi_i)
$$

所以这里的：

```python
.sum(1)
```

是在把每个动作分支的 entropy 合成一个样本级别的 entropy。

---

# 8. 对照修正你的注释

你原来的：

```python
# log_probs： 存储每一个动作的动作概率的对数值，shape 应该是（num_actions_branchs, B）
# entropies： 存储每一个动作分支的熵值，shape 应该是（num_actions_branchs, B）
```

更准确应该改成：

```python
# log_probs, entropies 这里一开始都是 list，不是 tensor
# list 长度 = num_action_branches
# 每个元素的 shape = (B,)
```

然后：

```python
log_probs = torch.stack(log_probs, dim=1)
```

之后：

```python
# log_probs shape = (B, num_action_branches)
```

以及：

```python
entropies = torch.stack(entropies, dim=1).sum(1).reshape(-1)
```

最终：

```python
# stack 后 shape = (B, num_action_branches)
# sum(1) 后 shape = (B,)
```

---

# 9. 当前这个项目里大多数时候为什么你不容易看出问题

因为这个项目里很多环境动作空间是：

```python
self.action_space_shape = (dummy_env.action_space.n,)
```

也就是通常只有 **1 个动作分支**。

这时：

- `num_action_branches = 1`
- `(B, 1)` 和 `(1, B)` 在很多直觉上容易混
- 但本质上仍然不同

所以注释里写错了，代码未必出错，只是因为当前多数场景动作分支数恰好是 1，不容易暴露。

---

# 10. 一句话总结

你的 shape 注释**不完全正确**，主要错在：

- `log_probs` / `entropies` 在 `stack` 前其实是列表，不宜直接说 shape
- 每个元素 shape 是 `(B,)`
- `torch.stack(..., dim=1)` 后的 shape 是：
  $$
  (B,\ num\_action\_branches)
  $$
  而不是：
  $$
  (num\_action\_branches,\ B)
  $$

如果你愿意，我可以继续帮你把这一段注释直接整理成一版“完全正确、适合学习用”的中文注释文本。