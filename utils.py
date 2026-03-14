import numpy as np
import torch

from torch import nn

from environments.cartpole_env import CartPole
from environments.memory_gym_env import MemoryGymWrapper
from environments.minigrid_env import Minigrid
from environments.poc_memory_env import PocMemoryEnv

def create_env(config:dict, render:bool=False):
    """Initializes an environment based on the provided environment name.
    
    Arguments:
        env_name {str}: Name of the to be instantiated environment 创建环境参数的一些配置
        render {bool}: Whether to instantiate the environment in render mode. (default: {False}) 是否开启界面显示

    Returns:
        {env}: Returns the selected environment instance. 返回创建号的环境实例
        todo 先看PocMemoryEnv 在看Minigrid

        这里创建环境，没有看到特殊的包装器，可能是因为自己
    """
    if config["type"] == "PocMemoryEnv":
        return PocMemoryEnv(glob=False, freeze=True, max_episode_steps=32)
    if config["type"] == "CartPole":
        return CartPole(mask_velocity=False)
    if config["type"] == "CartPoleMasked":
        return CartPole(mask_velocity=True)
    if config["type"] == "Minigrid":
        return Minigrid(config["name"])
    # todo 据说这里是将不同的环境封装为统一的调用环境接口，感觉这里是应该是将其他的游戏环境（"SearingSpotlights", "MortarMayhem", "MortarMayhem-Grid", "MysteryPath", "MysteryPath-Grid"）进行包装
    if config["type"] in ["SearingSpotlights", "MortarMayhem", "MortarMayhem-Grid", "MysteryPath", "MysteryPath-Grid"]:
        return MemoryGymWrapper(env_name = config["name"], reset_params=config["reset_params"], realtime_mode=render)

def polynomial_decay(initial:float, final:float, max_decay_steps:int, power:float, current_step:int) -> float:
    """Decays hyperparameters polynomially. If power is set to 1.0, the decay behaves linearly. 

    Arguments:
        initial {float} -- Initial hyperparameter such as the learning rate 初始值
        final {float} -- Final hyperparameter such as the learning rate 最终值
        max_decay_steps {int} -- The maximum numbers of steps to decay the hyperparameter 距离更新到最终值时的迭代次数
        power {float} -- The strength of the polynomial decay (e.g. 1.0 for linear decay) 多项式衰减的强度，1.0表示线性衰减
        current_step {int} -- The current step of the training 当前训练步数

    Returns:
        {float} -- Decayed hyperparameter 衰减后的超参数
    """
    # Return the final value if max_decay_steps is reached or the initial and the final value are equal
    if current_step > max_decay_steps or initial == final:
        return final
    # Return the polynomially decayed value given the current step
    else:
        return  ((initial - final) * ((1 - current_step / max_decay_steps) ** power) + final)
    
def batched_index_select(input, dim, index):
    """
    Selects values from the input tensor at the given indices along the given dimension.
    This function is similar to torch.index_select, but it supports batched indices.
    The input tensor is expected to be of shape (batch_size, ...), where ... means any number of additional dimensions.
    The indices tensor is expected to be of shape (batch_size, num_indices), where num_indices is the number of indices to select for each element in the batch.
    The output tensor is of shape (batch_size, num_indices, ...), where ... means any number of additional dimensions that were present in the input tensor.

    这里应该是类似根据index从input中第dim个维度获取对应的张量信息
    todo 在看具体的执行流程

    Arguments:
        input {torch.tensor} -- Input tensor 输入的tensor
        dim {int} -- Dimension along which to select values 输入待索引的维度
        index {torch.tensor} -- Tensor containing the indices to select 输入要提取的位置索引，每个 batch 样本自己的索引表。
        具体看md文档

    Returns:
        {torch.tensor} -- Output tensor
    """
    for ii in range(1, len(input.shape)):
        if ii != dim:
            # 这里是在给index增加维度，估计是为了后续能够索引对应的数据，gather应该是要让两者的维度相同
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    # 以下三步是为了将index的维度扩展到和input一样，才能进行gather操作，但是保留batch维度和索引维度不变
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def process_episode_info(episode_info:list) -> dict:
    """Extracts the mean and std of completed episode statistics like length and total reward.

    Arguments:
        episode_info {list} -- list of dictionaries containing results of completed episodes during the sampling phase

    Returns:
        {dict} -- Processed episode results (computes the mean and std for most available keys)
    """
    result = {}
    if len(episode_info) > 0:
        for key in episode_info[0].keys():
            if key == "success":
                # This concerns the PocMemoryEnv only
                episode_result = [info[key] for info in episode_info]
                result[key + "_percent"] = np.sum(episode_result) / len(episode_result)
            result[key + "_mean"] = np.mean([info[key] for info in episode_info])
            result[key + "_std"] = np.std([info[key] for info in episode_info])
    return result

class Module(nn.Module):
    """nn.Module is extended by functions to compute the norm and the mean of this module's parameters."""
    def __init__(self):
        super().__init__()

    def grad_norm(self):
        """Concatenates the gradient of this module's parameters and then computes the norm.

        Returns:
            {float}: Returns the norm of the gradients of this model's parameters. Returns None if no parameters are available.
        """
        grads = []
        for name, parameter in self.named_parameters():
            grads.append(parameter.grad.view(-1))
        return torch.linalg.norm(torch.cat(grads)).item() if len(grads) > 0 else None

    def grad_mean(self):
        """Concatenates the gradient of this module's parameters and then computes the mean.

        Returns:
            {float}: Returns the mean of the gradients of this module's parameters. Returns None if no parameters are available.
        """
        grads = []
        for name, parameter in self.named_parameters():
            grads.append(parameter.grad.view(-1))
        return torch.mean(torch.cat(grads)).item() if len(grads) > 0 else None