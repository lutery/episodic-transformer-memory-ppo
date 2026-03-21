import numpy as np
import os
import pickle
import time
import torch

from collections import deque
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from buffer import Buffer
from model import ActorCriticModel
from utils import batched_index_select, create_env, polynomial_decay, process_episode_info
from worker import Worker

class PPOTrainer:
    def __init__(self, config:dict, run_id:str="run", device:torch.device=torch.device("cpu")) -> None:
        """Initializes all needed training components.

        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            run_id {str, optional} -- A tag used to save Tensorboard Summaries and the trained model. Defaults to "run".
            device {torch.device, optional} -- Determines the training device. Defaults to cpu.
        """
        # Set members
        self.config = config # 训练的配置文件
        self.device = device # 运行的设备
        self.run_id = run_id # 本次训练的id
        self.num_workers = config["n_workers"] # 工作线程数量
        self.lr_schedule = config["learning_rate_schedule"] # 学习率调度
        self.beta_schedule = config["beta_schedule"] # beta调度 todo 用处
        self.cr_schedule = config["clip_range_schedule"] # 剪切范围调度 todo 用处
        self.memory_length = config["transformer"]["memory_length"] # 记忆长度，也就是模型一次性处理的序列长度
        self.num_blocks = config["transformer"]["num_blocks"] # transformer块数量
        self.embed_dim = config["transformer"]["embed_dim"] # 嵌入维度

        # Setup Tensorboard Summary Writer
        if not os.path.exists("./summaries"):
            os.makedirs("./summaries")
        timestamp = time.strftime("/%Y%m%d-%H%M%S" + "/")
        self.writer = SummaryWriter("./summaries/" + run_id + timestamp)

        # Init dummy environment to retrieve action space, observation space and max episode length
        # 在这里创建环境
        print("Step 1: Init dummy environment")
        dummy_env = create_env(self.config["environment"])
        observation_space = dummy_env.observation_space # 观察空间的obs 信息
        self.action_space_shape = (dummy_env.action_space.n,) # 离散动作的数量
        self.max_episode_length = dummy_env.max_episode_steps # 游戏的运行最大步数
        dummy_env.close() # 看来这里仅仅只是创建游戏来获取一些信息

        # Init buffer 构建缓冲区
        print("Step 2: Init buffer")
        self.buffer = Buffer(self.config, observation_space, self.action_space_shape, self.max_episode_length, self.device)

        # Init model 构建模型，内部包含动作策略模型和价值预测模型
        print("Step 3: Init model and optimizer")
        self.model = ActorCriticModel(self.config, observation_space, self.action_space_shape, self.max_episode_length).to(self.device)
        self.model.train()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr_schedule["initial"])

        # Init workers 构建环境工作线程
        # 多进程异步操作
        print("Step 4: Init environment workers")
        self.workers = [Worker(self.config["environment"]) for w in range(self.num_workers)]
        # 为每个环境创建id
        self.worker_ids = range(self.num_workers)
        # 看起来存储每个work进行步数 shape 是（num_workers，）每个元素表示对应worker当前的步数，每当worker完成一个episode时就重置为0
        # 主要是用在样本采集时，每个work的当前步数
        self.worker_current_episode_step = torch.zeros((self.num_workers, ), dtype=torch.long)
        # Reset workers (i.e. environments)
        # 初始化环境并获取第一帧观察
        print("Step 5: Reset workers")
        for worker in self.workers:
            worker.child.send(("reset", None))
        # Grab initial observations and store them in their respective placeholder location
        self.obs = np.zeros((self.num_workers,) + observation_space.shape, dtype=np.float32)
        for w, worker in enumerate(self.workers):
            self.obs[w] = worker.child.recv()

        # Setup placeholders for each worker's current episodic memory
        # 这里的memeory存储的是transformer中每个时间步的历史记忆项，用来和后续的最新观察组合预测最新的策略、价值
        # 注意这里的历史记忆是当前回合的历史记忆，一旦一个游戏结束后就会将memory存储到buffer中，并且重置当前work的memory为全零，开始新的回合的记忆积累
        self.memory = torch.zeros((self.num_workers, self.max_episode_length, self.num_blocks, self.embed_dim), dtype=torch.float32)
        # Generate episodic memory mask used in attention
        # 构建一个最大记忆长度的下三角矩阵
        self.memory_mask = torch.tril(torch.ones((self.memory_length, self.memory_length)), diagonal=-1)
        """ e.g. memory mask tensor looks like this if memory_length = 6
        0, 0, 0, 0, 0, 0
        1, 0, 0, 0, 0, 0
        1, 1, 0, 0, 0, 0
        1, 1, 1, 0, 0, 0
        1, 1, 1, 1, 0, 0
        1, 1, 1, 1, 1, 0
        """         
        # Setup memory window indices to support a sliding window over the episodic memory 提前构造好“每个时间步应该取哪一段 memory window”的索引表。
        # 具体看markdown
        repetitions = torch.repeat_interleave(torch.arange(0, self.memory_length).unsqueeze(0), self.memory_length - 1, dim = 0).long()
        self.memory_indices = torch.stack([torch.arange(i, i + self.memory_length) for i in range(self.max_episode_length - self.memory_length + 1)]).long()
        self.memory_indices = torch.cat((repetitions, self.memory_indices))
        """ e.g. the memory window indices tensor looks like this if memory_length = 4 and max_episode_length = 7:
        0, 1, 2, 3
        0, 1, 2, 3
        0, 1, 2, 3
        0, 1, 2, 3
        1, 2, 3, 4
        2, 3, 4, 5
        3, 4, 5, 6
        """

    def run_training(self) -> None:
        """Runs the entire training logic from sampling data to optimizing the model. Only the final model is saved."""
        # 这里是训练的主流程代码
        # 本函数就是开启正式训练
        print("Step 6: Starting training using " + str(self.device))
        # Store episode results for monitoring statistics
        episode_infos = deque(maxlen=100) # 看起来是仅存储近100轮游戏

        # 这个参数应该是训练几轮
        for update in range(self.config["updates"]):
            # Decay hyperparameters polynomially based on the provided config
            # 根据训练的轮数获取当前的学习率、beta、裁剪范围等
            learning_rate = polynomial_decay(self.lr_schedule["initial"], self.lr_schedule["final"], self.lr_schedule["max_decay_steps"], self.lr_schedule["power"], update)
            beta = polynomial_decay(self.beta_schedule["initial"], self.beta_schedule["final"], self.beta_schedule["max_decay_steps"], self.beta_schedule["power"], update)
            clip_range = polynomial_decay(self.cr_schedule["initial"], self.cr_schedule["final"], self.cr_schedule["max_decay_steps"], self.cr_schedule["power"], update)

            # Sample training data
            sampled_episode_info = self._sample_training_data()

            # Prepare the sampled data inside the buffer (splits data into sequences)
            self.buffer.prepare_batch_dict()

            # Train epochs
            training_stats, grad_info = self._train_epochs(learning_rate, clip_range, beta) # 返回监控的训练信息以及模型的梯度的信息
            training_stats = np.mean(training_stats, axis=0) # 计算所有监控信息的平均值

            # Store recent episode infos
            episode_infos.extend(sampled_episode_info)
            episode_result = process_episode_info(episode_infos)

            # Print training statistics
            # 其他他们打印的信息没什么太大区别，就是多了一个成功率
            if "success" in episode_result: # 这里是打印如果存在任意一次的游戏有成功完成时，则打印的信息
                result = "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} success={:.2f} pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}".format(
                    update, episode_result["reward_mean"], episode_result["reward_std"], episode_result["length_mean"], episode_result["length_std"], episode_result["success"],
                    training_stats[0], training_stats[1], training_stats[3], training_stats[2], torch.mean(self.buffer.values), torch.mean(self.buffer.advantages))
            else:
                # 如果不存在一次游戏成功完成，则不打印成功率
                result = "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}".format(
                    update, episode_result["reward_mean"], episode_result["reward_std"], episode_result["length_mean"], episode_result["length_std"], 
                    training_stats[0], training_stats[1], training_stats[3], training_stats[2], torch.mean(self.buffer.values), torch.mean(self.buffer.advantages))
            print(result)

            # Write training statistics to tensorboard 将监控的信息写入到tensorboard中
            self._write_gradient_summary(update, grad_info)
            self._write_training_summary(update, training_stats, episode_result)

        # Save the trained model at the end of the training
        # 完成训练，保存模型
        self._save_model()

    def _sample_training_data(self) -> list:
        """Runs all n workers for n steps to sample training data.
        从所有的子线程采集训练数据

        Returns:
            {list} -- list of results of completed episodes.
        """
        episode_infos = [] # 不区分是哪个work直接存储每次游戏结束时的info
        
        # Init episodic memory buffer using each workers' current episodic memory
        # todo 这是在干啥
        # 只能看出为每个采集work单独设立缓冲区
        # 利用索引，经过之前构建的memory_indices表，来为每个work设置对应的memory window索引
        self.buffer.memories = [self.memory[w] for w in range(self.num_workers)]
        # 这里应该是为每一个buffer设置workid
        for w in range(self.num_workers):
            self.buffer.memory_index[w] = w

        # Sample actions from the model and collect experiences for optimization
        for t in range(self.config["worker_steps"]): # 在每次训练前，最多执行多少次数据采集
            # Gradients can be omitted for sampling training data
            with torch.no_grad():
                # Store the initial observations inside the buffer 存储每步的观察
                self.buffer.obs[:, t] = torch.tensor(self.obs) 
                # Store mask and memory indices inside the buffer
                # torch.clip(self.worker_current_episode_step, 0, self.memory_length - 1)： 限制每个步数不能超过model的记忆长度，超过了就一直取最大记忆长度
                # 从memory_mask中提取对应步数的记忆掩码，存储在buffer中，确认当前能看到的范围
                self.buffer.memory_mask[:, t] = self.memory_mask[torch.clip(self.worker_current_episode_step, 0, self.memory_length - 1)]
                # 从memory_indices中提取对应步数的记忆索引，存储在buffer中，即在当前步中应该查看episode memory中的哪几个位置
                self.buffer.memory_indices[:, t] = self.memory_indices[self.worker_current_episode_step]
                # Retrieve the memory window from the entire episodic memory 根据索引提取对应的记忆
                sliced_memory = batched_index_select(self.memory, 1, self.buffer.memory_indices[:,t])
                # Forward the model to retrieve the policy, the states' value and the new memory item
                # 将当前观察，历史记忆，记忆的掩码，记忆的索引输入模型，获取动作策略、状态值函数、以及新的记忆项
                # 这里传入self.buffer.memory_indices是为了在位置编码时获取对应位置的位置编码
                policy, value, memory = self.model(torch.tensor(self.obs), sliced_memory, self.buffer.memory_mask[:, t],
                                                   self.buffer.memory_indices[:,t])
                
                # Add new memory item to the episodic memory 讲新的记忆项添加到历史记忆中
                self.memory[self.worker_ids, self.worker_current_episode_step] = memory

                # Sample actions from each individual policy branch
                actions = [] # 存储每个动作分支采样的动作
                log_probs = [] # 存储每个动作分支采样的动作对应的对数概率
                for action_branch in policy: # 对每一个动作分支进行遍历
                    action = action_branch.sample() # 采样动作
                    actions.append(action)
                    log_probs.append(action_branch.log_prob(action)) # 动作对应的对数概率
                # Write actions, log_probs and values to buffer 存储预测的动作、动作的对数概率、状态价值
                self.buffer.actions[:, t] = torch.stack(actions, dim=1) # 存储每个动作分支采样的动作
                self.buffer.log_probs[:, t] = torch.stack(log_probs, dim=1) # 存储每个动作分支采样的动作对应的对数概率
                self.buffer.values[:, t] = value # 存储状态价值

            # Send actions to the environments 将动作发送到环境中执行
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", self.buffer.actions[w, t].cpu().numpy()))

            # Retrieve step results from the environments
            # 接收执行的结果
            for w, worker in enumerate(self.workers):
                obs, self.buffer.rewards[w, t], self.buffer.dones[w, t], info = worker.child.recv()
                if info: # i.e. done
                    # 看来这里是如果有info代表游戏结束
                    # Reset the worker's current timestep
                    self.worker_current_episode_step[w] = 0 # 将对应work的当前步数设置为0
                    # Store the information of the completed episode (e.g. total reward, episode length)
                    episode_infos.append(info)
                    # Reset the agent (potential interface for providing reset parameters)
                    worker.child.send(("reset", None))
                    # Get data from reset
                    obs = worker.child.recv()
                    # Break the reference to the worker's memory
                    mem_index = self.buffer.memory_index[w, t] # 获取当前的采集的一个回合的观察记忆是对应哪个index，后续根据这个index到self.buffer.memories中找到对应的记忆
                    self.buffer.memories[mem_index] = self.buffer.memories[mem_index].clone() # 备份当前回合的记忆
                    # Reset episodic memory 由于游戏结束，则重置对应work的历史记忆，应该是将对应work的历史记忆清零
                    self.memory[w] = torch.zeros((self.max_episode_length, self.num_blocks, self.embed_dim), dtype=torch.float32)
                    if t < self.config["worker_steps"] - 1: # todo 这里啥时候可以达到或者超过self.config["worker_steps"] - 1:
                        # Store memory inside the buffer
                        self.buffer.memories.append(self.memory[w]) # 将构建的新的历史记忆缓冲区添加到memories，用于后续的使用
                        # Store the reference of to the current episodic memory inside the buffer
                        self.buffer.memory_index[w, t + 1:] = len(self.buffer.memories) - 1 # 更新buffer中对应work的memory index，指向新的memory
                else:
                    # Increment worker timestep
                    self.worker_current_episode_step[w] +=1 # 更新每个work当前的步数
                # Store latest observations
                self.obs[w] = obs # 这里是更新最新的obs
                            
        # Compute the last value of the current observation and memory window to compute GAE
        last_value = self.get_last_value() # 这里应该是计算最后一步的价值，避免因为假done导致模型误以为游戏结束了，导致最后一步的奖励和优势函数计算错误
        # Compute advantages
        self.buffer.calc_advantages(last_value, self.config["gamma"], self.config["lamda"])

        return episode_infos

    def get_last_value(self):
        """Returns:
                {torch.tensor} -- Last value of the current observation and memory window to compute GAE"""
        # 这里的start和end是一个batch，对应每个work当前所需要的start 和 end
        start = torch.clip(self.worker_current_episode_step - self.memory_length, 0)
        end = torch.clip(self.worker_current_episode_step, self.memory_length)
        indices = torch.stack([torch.arange(start[b],end[b]) for b in range(self.num_workers)]).long()
        # 根据索引选择对应的历史记忆，因为模型有一个记忆窗口
        sliced_memory = batched_index_select(self.memory, 1, indices) # Retrieve the memory window from the entire episode
        # 预测当前状态下的价值
        _, last_value, _ = self.model(torch.tensor(self.obs),
                                        sliced_memory, self.memory_mask[torch.clip(self.worker_current_episode_step, 0, self.memory_length - 1)],
                                        self.buffer.memory_indices[:,-1])
        return last_value

    def _train_epochs(self, learning_rate:float, clip_range:float, beta:float) -> list:
        """Trains several PPO epochs over one batch of data while dividing the batch into mini batches.
        采用PPO算法训练模型
        
        Arguments:
            learning_rate {float} -- The current learning rate 学习率
            clip_range {float} -- The current clip range 裁剪范围
            beta {float} -- The current entropy bonus coefficient beta值
            
        Returns:
            {tuple} -- Training and gradient statistics of one training epoch"""
        # grad_info 存储每次训练后的梯度变化信息
        train_info, grad_info = [], {}
        for _ in range(self.config["epochs"]):
            # 这里就是PPO提取每一次训练的batch
            mini_batch_generator = self.buffer.mini_batch_generator()
            for mini_batch in mini_batch_generator:
                # 又封装了一层进行小batch训练
                train_info.append(self._train_mini_batch(mini_batch, learning_rate, clip_range, beta))
                for key, value in self.model.get_grad_norm().items():
                    # 当前一次反向传播之后，模型各个模块参数梯度的范数大小
                    # 它是在看“这次更新里，每一层收到了多大的梯度信号”
                    grad_info.setdefault(key, []).append(value) 
        return train_info, grad_info

    def _train_mini_batch(self, samples:dict, learning_rate:float, clip_range:float, beta:float) -> list:
        """Uses one mini batch to optimize the model.

        Arguments:
            mini_batch {dict} -- The to be used mini batch data to optimize the model
            learning_rate {float} -- Current learning rate
            clip_range {float} -- Current clip range
            beta {float} -- Current entropy bonus coefficient

        Returns:
            {list} -- list of trainig statistics (e.g. loss)
        """
        # Select episodic memory windows
        # todo 搞清楚samples中每一个的来源 samples["memories"]是啥？
        # samples["memories"]：这条样本所属 episode 的完整 memory 张量，(B, max_episode_length, num_blocks, embed_dim)
        # samples["memory_indices"]：每条样本应该取哪几个时间位置的索引窗口，(B, memory_length)，例如某条样本可能是：[12, 13, 14, 15]，表示这条样本在训练时要查看 episode memory 中第 12 到 15 号位置
        # batched_index_select(..., 1, ...) 这里是在第 1 维做 batched gather。对 batch 里的每一条样本，都用它自己的 memory_indices，从自己的完整 episode memory 中取出对应的时间窗口。
        #   最终得到的 memory shape 大致是：(B, memory_length, num_blocks, embed_dim) 这个结果才是后面真正喂给模型的记忆窗口
        memory = batched_index_select(samples["memories"], 1, samples["memory_indices"])
        
        # Forward model
        # todo 查清楚最新的记忆是如何添加到缓冲区的
        policy, value, _ = self.model(samples["obs"], memory, samples["memory_mask"], samples["memory_indices"])

        # Retrieve and process log_probs from each policy branch
        # log_probs： 存储每一个动作的动作概率的对数值，shape 应该是（num_actions_branchs, B） todo 确认这边的shape是否有问题？为啥会算错
        # entropies： 存储每一个动作分支的熵值，shape 应该是（num_actions_branchs, B） todo
        log_probs, entropies = [], []
        for i, policy_branch in enumerate(policy):
            # samples["actions"][:, i]: 遍历采集样本的每一个动作分支 todo 看看具体是如何采集保存的？还是不是每一个分支而是本身就要遍历每一个离散动作？
            # 或者每一个执行动作的log概率
            log_probs.append(policy_branch.log_prob(samples["actions"][:, i]))
            entropies.append(policy_branch.entropy())
        log_probs = torch.stack(log_probs, dim=1) #
        entropies = torch.stack(entropies, dim=1).sum(1).reshape(-1)

        # Compute policy surrogates to establish the policy loss
        normalized_advantage = (samples["advantages"] - samples["advantages"].mean()) / (samples["advantages"].std() + 1e-8) # 类似ppo的归一化优势 shape is （B，）
        normalized_advantage = normalized_advantage.unsqueeze(1).repeat(1, len(self.action_space_shape)) # Repeat is necessary for multi-discrete action spaces t这里是针对多离散动作空间的处理，重复优势值以适配每一个动作分支，shape is （B，num_action_branches）
        # samples["log_probs"] 是采集动作时的动作概率的对数值，shape is （B，num_action_branches）
        # log_ratio 是当前策略和采集时策略的动作概率对数值之差，shape is （B，num_action_branches）
        log_ratio = log_probs - samples["log_probs"] # 计算新旧动作之间的概率对数值之差（实际上也比比率，因为是对数可以转换为减法），shape is （B，num_action_branches）
        ratio = torch.exp(log_ratio) # 去除对数，得到新旧动作概率的比率，shape is （B，num_action_branches）
        # 和普通的ppo一样，计算剪切版本的 surrogate loss 和 原始 surrogate loss，并取两者的最小值作为最终的策略损失，shape is （B，num_action_branches）
        surr1 = ratio * normalized_advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * normalized_advantage
        policy_loss = torch.min(surr1, surr2)
        policy_loss = policy_loss.mean()

        # Value  function loss
        sampled_return = samples["values"] + samples["advantages"] # todo 这里的values和我看过的ppo算法哪个部分比较相似？这里的sampled_return应该是要预测的回报值
        # 这里的值可以认为是预测的value，只是通过其他的方法实现不让预测的value过于偏离采集时的value，增加训练的稳定性
        clipped_value = samples["values"] + (value - samples["values"]).clamp(min=-clip_range, max=clip_range) 
        # 这里取max可以这里理解，一开始sampled_return和value很远，和clipped_value很近
        # 一开始更新的时候取value - sampled_return
        # 随着训练，value - sampled_return逐渐接近，离clipped_value - sampled_return越远，为了避免
        # 更新过大，然后如果一旦超过了裁剪范围，那么久重新让预测的值拉回到裁剪范围内（有可能也不会拉，因为一旦clap后，梯度就是0了），远离sampled_return的值，避免过拟合，提高鲁棒性
        # todo 这里不会是可重复使用历史训练记录的ppo吧？待排查
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = vf_loss.mean() # 训练预测价值损失

        # Entropy Bonus 计算最大熵奖励，鼓励策略的探索性，防止过早收敛到次优策略，这里是针对动作不要过拟合了
        entropy_bonus = entropies.mean()

        # Complete loss
        # 汇总所有的损失进行训练
        loss = -(policy_loss - self.config["value_loss_coefficient"] * vf_loss + beta * entropy_bonus)

        # Compute gradients
        # 卧槽，手动修改学习率
        for pg in self.optimizer.param_groups:
            pg["lr"] = learning_rate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["max_grad_norm"])
        self.optimizer.step() # 梯度更新参数

        # Monitor additional training stats
        approx_kl = (ratio - 1.0) - log_ratio # http://joschu.net/blog/kl-approx.html 计算kl散度的近似值，监控新旧策略之间的差距
        clip_fraction = (abs((ratio - 1.0)) > clip_range).float().mean() # 获取新旧动作之间被裁剪的比例，监控训练的稳定性，如果被裁减的多了，说明训练可能不稳定，更新步长过大

        return [policy_loss.cpu().data.numpy(), # 监控策略损失
                vf_loss.cpu().data.numpy(), # 监控价值损失
                loss.cpu().data.numpy(), # 监控总损失
                entropy_bonus.cpu().data.numpy(), # 监控熵奖励
                approx_kl.mean().cpu().data.numpy(), # 监控kl散度近似值
                clip_fraction.cpu().data.numpy()] # 监控被裁剪的动作比例

    def _write_training_summary(self, update, training_stats, episode_result) -> None:
        """Writes to an event file based on the run-id argument.

        Arguments:
            update {int} -- Current PPO Update
            training_stats {list} -- Statistics of the training algorithm
            episode_result {dict} -- Statistics of completed episodes
        """
        if episode_result:
            for key in episode_result:
                if "std" not in key:
                    self.writer.add_scalar("episode/" + key, episode_result[key], update)
        self.writer.add_scalar("losses/loss", training_stats[2], update)
        self.writer.add_scalar("losses/policy_loss", training_stats[0], update)
        self.writer.add_scalar("losses/value_loss", training_stats[1], update)
        self.writer.add_scalar("losses/entropy", training_stats[3], update)
        self.writer.add_scalar("training/value_mean", torch.mean(self.buffer.values), update)
        self.writer.add_scalar("training/advantage_mean", torch.mean(self.buffer.advantages), update)
        self.writer.add_scalar("other/clip_fraction", training_stats[4], update)
        self.writer.add_scalar("other/kl", training_stats[5], update)
        
    def _write_gradient_summary(self, update, grad_info):
        """Adds gradient statistics to the tensorboard event file.

        Arguments:
            update {int} -- Current PPO Update
            grad_info {dict} -- Gradient statistics
        """
        for key, value in grad_info.items():
            self.writer.add_scalar("gradients/" + key, np.mean(value), update)

    def _save_model(self) -> None:
        """Saves the model and the used training config to the models directory. The filename is based on the run id."""
        if not os.path.exists("./models"):
            os.makedirs("./models")
        self.model.cpu()
        pickle.dump((self.model.state_dict(), self.config), open("./models/" + self.run_id + ".nn", "wb"))
        print("Model saved to " + "./models/" + self.run_id + ".nn")

    def close(self) -> None:
        """Terminates the trainer and all related processes."""
        # 释放训练资源
        try:
            self.dummy_env.close()
        except:
            pass

        try:
            self.writer.close()
        except:
            pass

        try:
            for worker in self.workers:
                worker.child.send(("close", None))
        except:
            pass

        time.sleep(1.0)
        exit(0)