import torch
from docopt import docopt # 一个命令行参数解析库，作用是：直接根据帮助文档字符串解析 CLI 参数
from trainer import PPOTrainer
from yaml_parser import YamlParser

def main():
    # Command line arguments via docopt
    # 这里是命令行支持的参数定义格式，即是可以用来告诉用户如何调用，也可以用来根据这里的定义去解析参数
    # 如果不是这里面的参数则不会解析
    _USAGE = """
    Usage:
        train.py [options]
        train.py --help
    
    Options:
        --config=<path>            Path to the yaml config file [default: ./configs/poc_memory_env.yaml]
        --run-id=<path>            Specifies the tag for saving the tensorboard summary [default: run].
        --cpu                      Force training on CPU [default: False]
    """
    options = docopt(_USAGE) # 解析输入参数
    run_id = options["--run-id"] # 运行id号，每次实验建议不一样
    cpu = options["--cpu"] # 运行的设备
    # Parse the yaml config file. The result is a dictionary, which is passed to the trainer.
    # 参数套娃，在命令行中传入实际训练的参数yaml文件
    config = YamlParser(options["--config"]).get_config()

    # Determine the device to be used for training and set the default tensor type
    if not cpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            # 这里的意义：
            # 1. 设置默认的浮点类型的tensor
            # 2. 创建浮点类型的tensor时会默认创建在指定的设备上（仅对float有效）
            # 
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")

    # Initialize the PPO trainer and commence training
    # 构建训练器
    trainer = PPOTrainer(config, run_id=run_id, device=device)
    # 训练的主流程
    trainer.run_training()
    # 释放训练的资源
    trainer.close()

if __name__ == "__main__":
    main()