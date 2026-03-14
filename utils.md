# batched_index_select
`batched_index_select` 的作用一句话概括就是：

**对 batch 里的每个样本，用它自己的索引，在指定维度上分别取值。**

位置在 utils.py。

它解决的是一个普通 `torch.index_select` 不方便处理的问题：

> 不同 batch 样本，想取的索引不一样。

---

## 1. 它是干什么的

普通 `index_select` 更像是：

- 整个 tensor 所有 batch 共用同一套 index

而 `batched_index_select` 是：

- 第 0 个样本用自己的 index
- 第 1 个样本用自己的 index
- 第 2 个样本用自己的 index
- ...

所以它是一个：

**按 batch 分别索引的选择函数。**

---

## 2. 先看函数

在 utils.py：

```python
def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)
```

---

## 3. 参数分别是什么

### `input`
原始张量。

例如在 trainer 里常见的是：

- `self.memory`
- `samples["memories"]`

这种形状通常像：

$$
(B,\ T,\ ...)
$$

---

### `dim`
要在哪个维度上取值。

例如：

```python
dim = 1
```

表示在“时间维 / memory 维”上取。

---

### `index`
每个 batch 样本自己的索引表。

shape 一般是：

$$
(B,\ K)
$$

表示：
- batch 有 `B` 个样本
- 每个样本要取 `K` 个位置

例如：

```python
[
 [0, 1, 2, 3],
 [4, 5, 6, 7]
]
```

意思是：
- 第 0 个样本取位置 `[0,1,2,3]`
- 第 1 个样本取位置 `[4,5,6,7]`

---

# 4. 举一个最简单的例子

假设：

```python
input.shape = (2, 5)
input =
[
 [10, 11, 12, 13, 14],
 [20, 21, 22, 23, 24]
]
```

然后：

```python
index =
[
 [0, 2, 4],
 [1, 3, 4]
]
```

我们希望得到：

- 对第 0 行，取 `[0,2,4]` -> `[10,12,14]`
- 对第 1 行，取 `[1,3,4]` -> `[21,23,24]`

最终结果应该是：

```python
[
 [10, 12, 14],
 [21, 23, 24]
]
```

这就是 batched index select。

---

# 5. 它为什么不用 `index_select`

因为 `torch.index_select` 用同一套 index 作用于整批数据。  
比如它更像这样：

- 所有 batch 都取 `[0,2,4]`

但我们这里想要的是：

- 每个 batch 样本可以有不同 index

所以要用 `gather` 配合 reshape/expand 手动实现。

---

# 6. 逐行讲这段实现

---

## 第一步：给 `index` 补维度

```python
for ii in range(1, len(input.shape)):
    if ii != dim:
        index = index.unsqueeze(ii)
```

目的是让 `index` 的维度数和 `input` 对齐，方便后面 `expand` 和 `gather`。

比如如果：

```python
input.shape = (B, T, N, D)
dim = 1
index.shape = (B, K)
```

那这个循环之后，`index` 会变成类似：

```python
(B, K, 1, 1)
```

也就是除了 `dim=1` 那一维保留 `K`，其他非 batch 维都先补成单例维。

---

## 第二步：构造 expand 目标形状

```python
expanse = list(input.shape)
expanse[0] = -1
expanse[dim] = -1
```

比如：

```python
input.shape = (B, T, N, D)
```

如果 `dim=1`，那 `expanse` 就会变成：

```python
[-1, -1, N, D]
```

含义是：

- batch 维保留
- 目标索引维保留
- 其他维度扩展到和 input 一样

---

## 第三步：把 `index` 扩展成和 `input` 对齐的形状

```python
index = index.expand(expanse)
```

例如从：

```python
(B, K, 1, 1)
```

扩展到：

```python
(B, K, N, D)
```

这样它就能和 `input` 在除了被索引维以外的所有维度上对齐。

---

## 第四步：用 `gather` 真正取值

```python
return torch.gather(input, dim, index)
```

`gather` 的语义是：

> 在第 `dim` 维上，按 `index` 指定的位置逐元素取值。

因为现在 `index` 已经被扩展成和 `input` 匹配的形状，所以它就能实现“每个 batch 各取各的索引”。

---

# 7. 放到当前项目里怎么用

这个函数在项目里最重要的用途，是切 memory window。

例如在 trainer.py：

```python
sliced_memory = batched_index_select(self.memory, 1, self.buffer.memory_indices[:, t])
```

这里：

- `self.memory` shape 大致是  
  $$
  (n\_workers,\ max\_episode\_length,\ num\_blocks,\ embed\_dim)
  $$
- `dim=1`，表示沿着 episode 时间维取
- `self.buffer.memory_indices[:, t]` shape 是  
  $$
  (n\_workers,\ memory\_length)
  $$

意思就是：

**每个 worker 都按自己当前步对应的索引窗口，从整段 episode memory 里取出一段长度为 `memory_length` 的历史。**

得到的 `sliced_memory` shape 就是：

$$
(n\_workers,\ memory\_length,\ num\_blocks,\ embed\_dim)
$$

---

# 8. 再举一个项目内的直观例子

假设：

```python
self.memory.shape = (2, 7, 3, 8)
```

意思是：

- 2 个 worker
- 每个 worker 一整集最多 7 步 memory
- 每步存 3 个 block 的记忆
- 每个记忆向量 8 维

然后：

```python
index =
[
 [0, 1, 2, 3],
 [2, 3, 4, 5]
]
```

表示：

- worker 0 取前 4 步
- worker 1 取中间 4 步

那么：

```python
batched_index_select(self.memory, 1, index)
```

结果 shape 是：

```python
(2, 4, 3, 8)
```

即：

- 每个 worker 都取出了属于自己的 4 步 memory window

---

# 9. 为什么这个函数对 Transformer memory 很关键

因为当前项目不是每次只读一个 hidden state，而是每次要读一段历史窗口。

而不同 worker / 不同样本：

- 当前 episode 走到的步数不同
- 对应窗口位置不同

所以必须支持：

**同一个 batch 中，每条样本按各自不同的时间索引去切 memory。**

这就是 `batched_index_select` 存在的意义。

---

# 10. 一句话总结

`batched_index_select` 就是：

**在指定维度上，让 batch 中每个样本使用自己那一行索引，分别从输入张量中取出对应内容。**

在这个项目里，它最核心的用途是：

**从整段 episodic memory 中，为每个样本切出它当前需要的 memory window。**

如果你愿意，我可以下一条继续直接拿 trainer.py 那句，按真实 shape 从头到尾手动展开一遍。