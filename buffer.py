import numpy as np
import torch

from gym import spaces

class Buffer():
    """The buffer stores and prepares the training data. It supports transformer-based memory policies. """
    def __init__(self, config:dict, observation_space:spaces.Box, action_space_shape:tuple, max_episode_length:int, device:torch.device) -> None:
        """
        这里应该是游戏数据的缓存
        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model. 训练模型的一些超参数配置
            observation_space {spaces.Box} -- The observation space of the agent 观察空间的维度信息
            action_space_shape {tuple} -- Shape of the action space 动作的shape信息，不过在本代码中都是离散动作
            max_episode_length {int} -- The maximum number of steps in an episode 一轮游戏的最大步数
            device {torch.device} -- The device that will be used for training 
        """
        # Setup members
        self.device = device
        self.n_workers = config["n_workers"] # 这里的作用就是同时存储多少个环境的采集数据
        self.worker_steps = config["worker_steps"] # 每次采集的最大步数，也是每个小批量的步数，最终的批量大小是n_workers * worker_steps
        self.n_mini_batches = config["n_mini_batch"] # 小批量数量 todo
        self.batch_size = self.n_workers * self.worker_steps # 批量大小
        self.mini_batch_size = self.batch_size // self.n_mini_batches # 小批量大小
        self.max_episode_length = max_episode_length # 最大回合长度
        self.memory_length = config["transformer"]["memory_length"] # 记忆长度
        self.num_blocks = config["transformer"]["num_blocks"] # transformer块数量
        self.embed_dim = config["transformer"]["embed_dim"] # 嵌入维度

        # Initialize the buffer's data storage
        # 这里是采集数据的存储区域，第一个维度就是存储并列的环境数，第二个维度就是存储对应的数据
        self.rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        self.actions = torch.zeros((self.n_workers, self.worker_steps, len(action_space_shape)), dtype=torch.long)
        self.dones = np.zeros((self.n_workers, self.worker_steps), dtype=np.bool)
        self.obs = torch.zeros((self.n_workers, self.worker_steps) + observation_space.shape)
        self.log_probs = torch.zeros((self.n_workers, self.worker_steps, len(action_space_shape))) # 存储self.actions 对应动作的概率的log值
        self.values = torch.zeros((self.n_workers, self.worker_steps)) # 状态值函数
        self.advantages = torch.zeros((self.n_workers, self.worker_steps)) # 优势函数
        # Episodic memory index buffer
        # Whole episode memories
        # The length of memories is equal to the number of sampled episodes during training data sampling
        # Each element is of shape (max_episode_length, num_blocks, embed_dim)
        self.memories = [] # 存储每轮游戏的transformer记忆，每个元素是一个三维张量，第一维是最大回合长度，第二维是transformer块数量，第三维是嵌入维度 ，不断的累积，直到buffer定义的最大长度
        # Memory mask used during attention 存储的是每个步数对应的记忆掩码，确认当前能看到的范围
        # 这里的memory_mask是一个三维的布尔张量，第一维是并列的环境数，第二维是每个环境的步数，第三维是记忆长度（存储记忆掩码）。这个张量用于在注意力机制中掩盖掉当前步数无法看到的记忆部分。
        self.memory_mask = torch.zeros((self.n_workers, self.worker_steps, self.memory_length), dtype=torch.bool)
        # Index to select the correct episode memory from self.memories
        # 这个存储的是对应每个采集的步数（采集的步数对应config["n_workers"]）历史记忆的索引位置，这个索引位置对应self.memories中存储的历史记忆的位置，后续在训练过程中根据这个索引位置找到对应的历史记忆进行训练
        self.memory_index = torch.zeros((self.n_workers, self.worker_steps), dtype=torch.long)
        # Indices to slice the memory window 存储每个步数对应的记忆窗口索引，即在当前步中应该查看episode memory中的哪几个位置，也类似于步数
        # 这里的memory_indices是一个三维的长整型张量，第一维是并列的环境数，第二维是每个环境的步数，第三维是记忆长度（存储记忆窗口索引）。这个张量用于在训练过程中为每个步数指定对应的记忆窗口索引，以便模型能够正确地访问和利用历史信息。
        self.memory_indices = torch.zeros((self.n_workers, self.worker_steps, self.memory_length), dtype=torch.long)

    def prepare_batch_dict(self) -> None:
        """Flattens the training samples and stores them inside a dictionary. Due to using a recurrent policy,
        the data is split into episodes or sequences beforehand.
        """
        # Supply training samples
        samples = {
            "actions": self.actions,
            "values": self.values,
            "log_probs": self.log_probs,
            "advantages": self.advantages,
            "obs": self.obs,
            "memory_mask": self.memory_mask,
            "memory_index": self.memory_index,
            "memory_indices": self.memory_indices,
        }
        # Convert the memories to a tensor
        self.memories = torch.stack(self.memories, dim=0)

        # Flatten all samples and convert them to a tensor except memories and its memory mask
        self.samples_flat = {}
        for key, value in samples.items():
            self.samples_flat[key] = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])

    def mini_batch_generator(self):
        """A generator that returns a dictionary containing the data of a whole minibatch.
        This mini batch is completely shuffled.
            
        Yields:
            {dict} -- Mini batch data for training
        """
        # Prepare indices (shuffle)
        indices = torch.randperm(self.batch_size)
        mini_batch_size = self.batch_size // self.n_mini_batches
        for start in range(0, self.batch_size, mini_batch_size):
            # Compose mini batches
            end = start + mini_batch_size
            mini_batch_indices = indices[start: end]
            mini_batch = {}
            for key, value in self.samples_flat.items():
                if key == "memory_index":
                    # Add the correct episode memories to the concerned mini batch
                    mini_batch["memories"] = self.memories[value[mini_batch_indices]]
                else:
                    mini_batch[key] = value[mini_batch_indices].to(self.device)
            yield mini_batch

    def calc_advantages(self, last_value:torch.tensor, gamma:float, lamda:float) -> None:
        """Generalized advantage estimation (GAE)

        Arguments:
            last_value {torch.tensor} -- Value of the last agent's state 采集结束时最后一个状态的价值
            gamma {float} -- Discount factor 折扣因子
            lamda {float} -- GAE regularization parameter GAE正则化参数 
        """
        with torch.no_grad():
            last_advantage = 0
            mask = torch.tensor(self.dones).logical_not() # mask values on terminal states
            rewards = torch.tensor(self.rewards)
            for t in reversed(range(self.worker_steps)): # 这里是从后往前计算优势函数，最后一步的优势函数就是delta，前面每一步的优势函数都是当前的delta加上后一步的优势函数乘以折扣因子和GAE正则化参数
                last_value = last_value * mask[:, t] # 计算优势函数时，如果当前步是done了的状态，则last_value应该被置零，因为后续的奖励不应该被考虑进来
                last_advantage = last_advantage * mask[:, t] # 计算优势函数时，如果当前步是done了的状态，则last_advantage应该被置零，因为后续的奖励不应该被考虑进来
                delta = rewards[:, t] + gamma * last_value - self.values[:, t] # 这里的delta是当前步的奖励加上折扣后的后一步的价值减去当前步的价值，代表当前步的优势函数的一个估计
                last_advantage = delta + gamma * lamda * last_advantage # 这里的last_advantage是当前步的优势函数的一个估计，等于当前步的delta加上折扣后的后一步的优势函数乘以GAE正则化参数
                self.advantages[:, t] = last_advantage # 将当前步的优势函数估计存储到优势函数的缓存中
                last_value = self.values[:, t] # 更新last_value为当前步的价值，为前一步的优势函数计算做准备