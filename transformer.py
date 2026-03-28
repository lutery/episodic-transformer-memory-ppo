import numpy as np
import torch

from einops import rearrange
from torch import nn
from utils import Module

class MultiHeadAttention(nn.Module):
    """Multi Head Attention without dropout inspired by https://github.com/aladdinpersson/Machine-Learning-Collection
    https://youtu.be/U0s0f995w14"""
    def __init__(self, embed_dim, num_heads):
        """
        Arguments:
            embed_dim {int} -- Size of the embedding dimension 嵌入层的维度
            num_heads {int} -- Number of attention heads 注意力头的数量
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads # 每个头的维度

        assert (
            self.head_size * num_heads == embed_dim
        ), "Embedding dimension needs to be divisible by the number of heads"
    
        # 构建q\k\v
        self.values = nn.Linear(embed_dim, embed_dim, bias=False)
        self.keys = nn.Linear(embed_dim, embed_dim, bias=False)
        self.queries = nn.Linear(embed_dim, embed_dim, bias=False)
        # 将选择的结果进一步的提取总结特征
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, values, keys, queries, mask):
        """
        The forward pass of the multi head attention layer.
        
        Arguments:
            values {torch.tensor} -- Value in shape of (N, L, D) L = memory length 历史记忆的长度
            keys {torch.tensor} -- Keys in shape of (N, L, D) L = memory length 历史记忆的长度
            queries {torch.tensor} -- Queries in shape of (N, 1, D) 当前时间步的特征
            mask {torch.tensor} -- Attention mask in shape of (N, L) 掩码矩阵，用来屏蔽掉不需要关注的部分，比如未来信息
            
        Returns:
            torch.tensor -- Output
            torch.tensor -- Attention weights
        """
        # Get number of training examples and sequence lengths
        N = queries.shape[0] # batch size 训练的采样数量
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split the embedding into self.num_heads different pieces
        # 进一步贴图qkv各自关注的特征
        values = self.values(values)    # (N, value_len, embed_dim)
        keys = self.keys(keys)          # (N, key_len, embed_dim)
        queries = self.queries(queries) # (N, query_len, embed_dim)
        
        # 将qkv的头按维度划分
        values = values.reshape(N, value_len, self.num_heads, self.head_size)   # (N, value_len, heads, head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_size)         # (N, key_len, heads, head_dim)
        queries = queries.reshape(N, query_len, self.num_heads, self.head_size) # (N, query_len, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # 一次性完成qk的维度调整，相乘的操作，输出的energy shape is (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their attention weights become 0
        if mask is not None:
            # 屏蔽未来的信息
            energy = energy.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float("-1e20")) # -inf causes NaN

        # Normalize energy values and apply softmax wo retreive the attention scores
        # 转换为注意力分数，其中的除法操作主要是用来避免概率爆炸
        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        # Scale values by attention weights
        # 再次使用torch.einsum进行维度调整、然后相乘的操作
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_size
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        # Forward projection
        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_dim)

        # (N, query_len, embed_dim)
        # (N, heads, query_len, key_len)
        return out, attention
        
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, config):
        """Transformer Block made of LayerNorms, Multi Head Attention and one fully connected feed forward projection.
        Arguments:
            embed_dim {int} -- Size of the embeddding dimension 嵌入层的维度
            num_heads {int} -- Number of attention headds。注意力头的数量
            config {dict} -- General config 训练的配置文件的字典类
        """
        super(TransformerBlock, self).__init__()

        # Attention
        self.attention = MultiHeadAttention(embed_dim, num_heads)

        # Setup GTrXL if used
        self.use_gtrxl = config["gtrxl"] if "gtrxl" in config else False
        if self.use_gtrxl:
            # 这里的关键，使用了GTrXL
            self.gate1 = GRUGate(embed_dim, config["gtrxl_bias"])
            self.gate2 = GRUGate(embed_dim, config["gtrxl_bias"])

        # LayerNorms
        self.layer_norm = config["layer_norm"]
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        if self.layer_norm == "pre":
            # 这里的归一化是在输入的时候进行一次归一化
            self.norm_kv = nn.LayerNorm(embed_dim)

        # Feed forward projection 再次提取特征
        self.fc = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())

    def forward(self, value, key, query, mask):
        """
        Arguments:
            values {torch.tensor} -- Value in shape of (N, L, D) L = memory length 历史记忆的长度
            keys {torch.tensor} -- Keys in shape of (N, L, D) L = memory length 历史记忆的长度
            query {torch.tensor} -- Queries in shape of (N, L, D) 在本例中 L = 1
            mask {torch.tensor} -- Attention mask in shape of (N, L)
        Returns:
            torch.tensor -- Output
            torch.tensor -- Attention weights
        """
        # Apply pre-layer norm across the attention input
        # 对输入做归一化
        if self.layer_norm == "pre":
            query_ = self.norm1(query)
            value = self.norm_kv(value)
            key = value
        else:
            query_ = query

        # Forward MultiHeadAttention
        attention, attention_weights = self.attention(value, key, query_, mask)
        # attention shape is (N, query_len, embed_dim)
        # attentin_weights shape is (N, heads, query_len, key_len)

        # GRU Gate or skip connection
        # 用“可学习门控的残差连接”替代普通的直接相加残差。个项目里 GTrXL 的关键改动
        if self.use_gtrxl:
            # Forward GRU gating 当前表示要不要吸收这次 attention 读回来的内容？吸收多少？
            # 因为 memory 读回来的信息不一定总是有用。
            h = self.gate1(query, attention)
        else:
            # Skip connection
            h = attention + query
        
        # Apply post-layer norm across the attention output (i.e. projection input)
        if self.layer_norm == "post":
            h = self.norm1(h)

        # Apply pre-layer norm across the projection input (i.e. attention output)
        if self.layer_norm == "pre":
            h_ = self.norm2(h)
        else:
            h_ = h

        # Forward projection
        forward = self.fc(h_)

        # GRU Gate or skip connection
        if self.use_gtrxl:
            # Forward GRU gating FFN 的新变换要不要强烈覆盖当前状态？还是保守一点？
            out = self.gate2(h, forward)
        else:
            # Skip connection
            out = forward + h
        
        # Apply post-layer norm across the projection output
        if self.layer_norm == "post":
            out = self.norm2(out)

        return out, attention_weights

class SinusoidalPosition(nn.Module):
    """Relative positional encoding"""
    def __init__(self, dim, min_timescale = 2., max_timescale = 1e4):
        '''
        dim: 位置的嵌入维度
        min_timescale: 最小时间尺度
        max_timescale: 最大时间尺度
        '''
        super().__init__()
        freqs = torch.arange(0, dim, min_timescale) # shape (dim / min_timescale)
        inv_freqs = max_timescale ** (-freqs / dim) # inv_freqs shape is (dim / min_timescale)
        self.register_buffer('inv_freqs', inv_freqs) 

    def forward(self, seq_len):
        seq = torch.arange(seq_len - 1, -1, -1.)
        sinusoidal_inp = rearrange(seq, 'n -> n ()') * rearrange(self.inv_freqs, 'd -> () d')
        pos_emb = torch.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), dim = -1)
        return pos_emb # shape is (seq_len, dim)

class Transformer(nn.Module):
    """Transformer encoder architecture without dropout. Positional encoding can be either "relative", "learned" or "" (none)."""
    def __init__(self, config, input_dim, max_episode_steps) -> None:
        """Sets up the input embedding, positional encoding and the transformer blocks.
        Arguments:
            config {dict} -- Transformer config 配置文件中transformer的超参数
            input_dim {int} -- Dimension of the input 输入的维度
            max_episode_steps {int} -- Maximum number of steps in an episode 最大的步数
        """
        super().__init__()
        self.config = config
        self.num_blocks = config["num_blocks"] # 有几个Transformer Block
        self.embed_dim = config["embed_dim"] # 嵌入层的维度，在本代码中等于 input_dim
        self.num_heads = config["num_heads"] # 注意力头的数量
        self.max_episode_steps = max_episode_steps # 游戏的最大步数
        self.activation = nn.ReLU()

        # Input embedding layer 这个层是用来做什么的？对应Transformer中的哪一层
        # 对输入的特征进行进一步的特征提取，然后再送入transformer中
        self.linear_embedding = nn.Linear(input_dim, self.embed_dim)
        nn.init.orthogonal_(self.linear_embedding.weight, np.sqrt(2))

        # Determine positional encoding 
        if config["positional_encoding"] == "relative":
            # 这里应该是构建一个相对位置编码器
            self.pos_embedding = SinusoidalPosition(dim = self.embed_dim)
        elif config["positional_encoding"] == "learned":
            # 这里应该是构建一个可以训练学习的位置编码器，第一维是最大的步长，第二维是位置嵌入层的维度
            self.pos_embedding = nn.Parameter(torch.randn(self.max_episode_steps, self.embed_dim)) # (batch size, max episoded steps, num layers, layer size)
        else:
            pass    # No positional encoding is used
        
        # Instantiate transformer blocks 构建transformer编码器模块
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, config) 
            for _ in range(self.num_blocks)])

    def forward(self, h, memories, mask, memory_indices):
        """
        Arguments:
            h {torch.tensor} -- Input (query) 当前的观察
            memories {torch.tesnor} -- Whole episoded memories of shape (N, L, num blocks, D) 历史记忆
            mask {torch.tensor} -- Attention mask (dtype: bool) of shape (N, L) 观察掩码，可能是用于最开始的几步时候看不到未来
            memory_indices {torch.tensor} -- Memory window indices (dtype: long) of shape (N, L) 这个应该是序列的位置索引，用来说明当前应该是哪个时间步的位置，从位置编码器中，根据这个位置索引来获取对应的位置信息，然后加到记忆中去
        Returns:
            {torch.tensor} -- Output of the entire transformer encoder
            {torch.tensor} -- Out memories (i.e. inputs to the transformer blocks)
        """
        # Feed embedding layer and activate 将输入的当前观察转换为嵌入的维度
        h = self.activation(self.linear_embedding(h))

        # Add positional encoding to every transformer block input
        # 根据不同的位置编码
        # 为什么没给h添加位置编码看md
        if self.config["positional_encoding"] == "relative":
            pos_embedding = self.pos_embedding(self.max_episode_steps)[memory_indices]
            memories = memories + pos_embedding.unsqueeze(2) # 将位置编码和记忆结合在一起，感觉这里每次都增加是因为相对位置会随着观察的位置而不同
            # memories[:,:,0] = memories[:,:,0] + pos_embedding # add positional encoding only to first layer?
        elif self.config["positional_encoding"] == "learned":
            memories = memories + self.pos_embedding[memory_indices].unsqueeze(2) # 可学习的位置编码就没这么多事情了，直接增加进去
            # memories[:,:,0] = memories[:,:,0] + self.pos_embedding[memory_indices] # add positional encoding only to first layer?

        # Forward transformer blocks
        out_memories = [] # 这里存储的是每一层输入的最新的记忆
        # 遍历每一个block
        # h shape is (N, D)
        # 在memories中这里保存的 memory 不是一份，而是“每个 block 各存一份”。
        # 看来这里是手动保存每一层的历史记忆
        # 最终预测输出 h
        for i, block in enumerate(self.transformer_blocks):
            # 每层 block 的输入表示，作为这一层未来时间步可访问的 memory
            '''
            把当前 h 存进 memory 时切断计算图。

            这样未来时间步使用这些 memory 时，不会把梯度反向传回整个历史 episode。

            这是必要的，因为这里不是做完整 BPTT，而是把 episodic memory 当作一种缓存机制。

            否则：

            显存会爆
            计算图会跨很多时间步越来越大

            '''
            out_memories.append(h.detach()) # 这里用detach时保证存储起来的tensor时没有梯度的，避免将反向传播梯度影响到缓冲区
            # memories[:, :, i] shape is (N, L, D)
            h, attention_weights = block(memories[:, :, i], memories[:, :, i], h.unsqueeze(1), mask) # args: value, key, query, mask
            h = h.squeeze()
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
        # 
        # torch.stack(out_memories, dim=1)：(N, num_blocks, D)
        return h, torch.stack(out_memories, dim=1)
    
class GRUGate(nn.Module):
    """
    Overview:
    这里是GTrXL的关键代码
    一个门控单元决定：

    保留多少旧信息
    接收多少新信息
        GRU Gating Unit used in GTrXL.
        Inspired by https://github.com/dhruvramani/Transformers-RL/blob/master/layers.py
    """

    def __init__(self, input_dim: int, bg: float = 0.0):
        """
        Arguments:
            input_dim {int} -- Input dimension
            bg {float} -- Initial gate bias value. By setting bg > 0 we can explicitly initialize the gating mechanism to
            be close to the identity map. This can greatly improve the learning speed and stability since it
            initializes the agent close to a Markovian policy (ignore attention at the beginning). (default: {0.0})
        """
        super(GRUGate, self).__init__()
        # 这里是构建GRU门控单元的线性层
        self.Wr = nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = nn.Linear(input_dim, input_dim, bias=False)
        self.bg = nn.Parameter(torch.full([input_dim], bg))  # bias
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # 这边是对权重进行初始化
        nn.init.xavier_uniform_(self.Wr.weight)
        nn.init.xavier_uniform_(self.Ur.weight)
        nn.init.xavier_uniform_(self.Wz.weight)
        nn.init.xavier_uniform_(self.Uz.weight)
        nn.init.xavier_uniform_(self.Wg.weight)
        nn.init.xavier_uniform_(self.Ug.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """       
        两个待残差连接的张量 
        Arguments:
            x {torch.tensor} -- First input
            y {torch.tensor} -- Second input
        Returns:
            {torch.tensor} -- Output

            果 bg 比较大，那么一开始：

        Wz(y) + Uz(x) - bg 会偏小
        sigmoid(...) 会更小
        z 更接近 0
        这会导致输出更接近：

        (
        1
        −
        z
        )
        ⊙
        x
        +
        z
        ⊙
        h
        ≈
        x
        (1−z)⊙x+z⊙h≈x
        也就是：

        初始时更像 identity mapping。

        直观理解：

        一开始先少改动当前表示，别太依赖 attention / FFN；等训练稳定后，再逐渐学会打开门。
        """
        # x shape is (N, query_len, embed_dim) 旧信息
        # y shape is (N, query_len, embed_dim) 新信息
        # 具体看markdown
        r = self.sigmoid(self.Wr(y) + self.Ur(x)) # r shape is (N, query_len, embed_dim) 重置门，在生成候选状态时，旧信息 x 有多少要被带进去
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg) # z shape is (N, query_len, embed_dim) 更新门，决定多少新信息被引入，决定最终输出里：多少用旧信息 x、多少用新候选 h
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x))) # torch.mul(r, x)) shape is (N, query_len, embed_dim), h shape is (N, query_len, embed_dim) 候选状态，这是在构造一个“如果我要更新，那我更新成什么”的候选结果。
        return torch.mul(1 - z, x) + torch.mul(z, h)
