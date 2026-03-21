import numpy as np
import torch

from torch.distributions import Categorical
from torch import nn
from torch.nn import functional as F

from transformer import Transformer

class ActorCriticModel(nn.Module):
    def __init__(self, config, observation_space, action_space_shape, max_episode_length):
        """Model setup

        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model. 训练用的各种超参数
            observation_space {box} -- Properties of the agent's observation space
            action_space_shape {tuple} -- Dimensions of the action space
            max_episode_length {int} -- The maximum number of steps in an episode
        """
        super().__init__()
        self.hidden_size = config["hidden_layer_size"] # 隐藏层的维度
        self.memory_layer_size = config["transformer"]["embed_dim"] # 记忆层的维度，也是transformer块的输入输出维度
        self.observation_space_shape = observation_space.shape # 观察空间的shape 应该是用在输入尺寸上，主要是用来判断输入的obs是图像还是向量
        self.max_episode_length = max_episode_length

        # Observation encoder 这里是针对不同的输入，比如图像和一维的数据
        if len(self.observation_space_shape) > 1:
            # Case: visual observation is available
            # Visual encoder made of 3 convolutional layers
            # 如果是可视化图像，则将图像的特征压缩为一维特征
            self.conv1 = nn.Conv2d(observation_space.shape[0], 32, 8, 4,)
            self.conv2 = nn.Conv2d(32, 64, 4, 2, 0)
            self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
            nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))
            nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))
            nn.init.orthogonal_(self.conv3.weight, np.sqrt(2))
            # Compute output size of convolutional layers
            self.conv_out_size = self.get_conv_output(observation_space.shape)
            in_features_next_layer = self.conv_out_size
        else:
            # Case: vector observation is available
            in_features_next_layer = observation_space.shape[0]
        # 一个很大的疑问，它是怎么将样本作为序列化数据传入的处理的
        # 因为这里的处理仅仅只是针对当前的obs 状态，对于历史的状态则直接从forward中的memory的参数传入
        # 然后将forward后的memory输出作为新的历史记忆存储到buffer中，供后续训练使用
        
        # Hidden layer
        self.lin_hidden = nn.Linear(in_features_next_layer, self.memory_layer_size)
        nn.init.orthogonal_(self.lin_hidden.weight, np.sqrt(2))

        # Transformer Blocks 构建transformer 应该是encoder
        self.transformer = Transformer(config["transformer"], self.memory_layer_size, self.max_episode_length)

        # Decouple policy from value
        # Hidden layer of the policy 这里应该是进一步将特征提取为策略的特征
        self.lin_policy = nn.Linear(self.memory_layer_size, self.hidden_size)
        # orthogonal_ 是正交初始化，将矩阵初始化为一个正交矩阵
        # 不会随便把向量拉爆
        # 也不会轻易把向量压得很小
        # 更像“旋转 / 反射 / 保持尺度”
        # 跟 Xavier 最大的区别
            # 每个元素独立随机采样
            # 只控制整体方差范围
        nn.init.orthogonal_(self.lin_policy.weight, np.sqrt(2))

        # Hidden layer of the value function 这个层的作用是啥？价值预测分支，用于在最后一步预测价值前进一步的特征提取
        self.lin_value = nn.Linear(self.memory_layer_size, self.hidden_size)
        nn.init.orthogonal_(self.lin_value.weight, np.sqrt(2))

        # Outputs / Model heads
        # Policy (Multi-discrete categorical distribution)
        # 动作策略的预测分支，每个分支预测一组动作
        # 怎么使用，如果一个环境多个离散动作分支，则进行多个离散动作分支的预测
        self.policy_branches = nn.ModuleList()
        # 看起来还涉及到多组组合动作
        for num_actions in action_space_shape:
            actor_branch = nn.Linear(in_features=self.hidden_size, out_features=num_actions)
            nn.init.orthogonal_(actor_branch.weight, np.sqrt(0.01))
            self.policy_branches.append(actor_branch)
            
        # Value function 价值预测网络
        self.value = nn.Linear(self.hidden_size, 1)
        nn.init.orthogonal_(self.value.weight, 1)

    def forward(self, obs:torch.tensor, memory:torch.tensor, memory_mask:torch.tensor, memory_indices:torch.tensor):
        """Forward pass of the model

        Arguments:
            obs {torch.tensor} -- Batch of observations 当前的观察
            memory {torch.tensor} -- Episodic memory window 历史记忆，如果缺少的历史观察就用0填充
            memory_mask {torch.tensor} -- Mask to prevent the model from attending to the padding 观察掩码，可能是用于最开始的几步时候看不到未来
            memory_indices {torch.tensor} -- Indices to select the positional encoding that matches the memory window 表示在当前的步数下，能够看到的历史记忆的位置索引，可能是用于最开始的几步时候看不到未来

        Returns:
            {Categorical} -- Policy: Categorical distribution
            {torch.tensor} -- Value function: Value
        """
        # Set observation as input to the model
        h = obs
        # Forward observation encoder 提取图片观察的特征后，展平为一维的特征
        if len(self.observation_space_shape) > 1:
            batch_size = h.size()[0]
            # Propagate input through the visual encoder
            h = F.relu(self.conv1(h))
            h = F.relu(self.conv2(h))
            h = F.relu(self.conv3(h))
            # Flatten the output of the convolutional layers
            h = h.reshape((batch_size, -1))

        # Feed hidden layer 将特征的维度转换为记忆层的维度
        h = F.relu(self.lin_hidden(h))
        
        # Forward transformer blocks
        # 将特征输入到transformer块中，更新记忆，提取特征
        # h输出最新的记忆特征， memeory保存每一transformer层输入的记忆，用于后续存储到历史记忆中
        h, memory = self.transformer(h, memory, memory_mask, memory_indices)

        # Decouple policy from value
        # 下面就是对最新的特征h进行预测动作策略、状态价值
        # Feed hidden layer (policy)
        h_policy = F.relu(self.lin_policy(h))
        # Feed hidden layer (value function)
        h_value = F.relu(self.lin_value(h))
        # Head: Value function 预测价值
        value = self.value(h_value).reshape(-1)
        # Head: Policy 对每一个动作策略分支预测动作概率分布
        pi = [Categorical(logits=branch(h_policy)) for branch in self.policy_branches]
        
        # 返回动作策略、状态价值以及最新的记忆特征
        return pi, value, memory

    def get_conv_output(self, shape:tuple) -> int:
        """Computes the output size of the convolutional layers by feeding a dummy tensor.

        Arguments:
            shape {tuple} -- Input shape of the data feeding the first convolutional layer

        Returns:
            {int} -- Number of output features returned by the utilized convolutional layers
        """
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))
    
    def get_grad_norm(self):
        """Returns the norm of the gradients of the model.
        
        Returns:
            {dict} -- Dictionary of gradient norms grouped by layer name
        """
        grads = {} # 分别计算每一个层级的梯度范数，方便后续监控和分析
        if len(self.observation_space_shape) > 1: # 如果是图像状态，那么就会有这几个卷积的梯度
            grads["encoder"] = self._calc_grad_norm(self.conv1, self.conv2, self.conv3)  
            
        grads["linear_layer"] = self._calc_grad_norm(self.lin_hidden)
        
        transfomer_blocks = self.transformer.transformer_blocks
        for i, block in enumerate(transfomer_blocks):
            grads["transformer_block_" + str(i)] = self._calc_grad_norm(block)
        
        for i, head in enumerate(self.policy_branches):
            grads["policy_head_" + str(i)] = self._calc_grad_norm(head)
        
        grads["lin_policy"] = self._calc_grad_norm(self.lin_policy)
        grads["value"] = self._calc_grad_norm(self.lin_value, self.value)
        grads["model"] = self._calc_grad_norm(self, self.value)
          
        return grads
    
    def _calc_grad_norm(self, *modules):
        """Computes the norm of the gradients of the given modules.
        一个通用的获取输入的module的梯度范数的函数，主要是为了方便获取不同层的梯度范数，进行监控和分析
        在这里计算的因为它会把所有元素综合起来看，而不是只看某一个值，所以更能反映整体的梯度情况，而不是某一个元素的情况

        Arguments:
            modules {list} -- List of modules to compute the norm of the gradients of.

        Returns:
            {float} -- Norm of the gradients of the given modules. 
        """
        grads = []
        for module in modules:
            for name, parameter in module.named_parameters():
                # 把每个参数张量的梯度拉平成一维
                # 
                grads.append(parameter.grad.view(-1))
        # 把所有参数梯度拼成一个超长向量，然后计算这个向量的范数，反映整体的梯度情况
        return torch.linalg.norm(torch.cat(grads)).item() if len(grads) > 0 else None