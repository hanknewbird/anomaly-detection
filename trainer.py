from ignite.engine import Engine, Events
from ignite.metrics import Loss, RunningAverage
from tensorboardX import SummaryWriter
import torch
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
import os


def create_summary_writer(model, train_loader, log_dir, save_graph, device):
    """创建一个tensorboard数据

    Arguments:
        model                     -- 需要保存其图形的模型
        train_loader {dataloader} -- 训练集加载器
        log_dir {str}             -- 日志路径
        save_graph {bool}         -- 如果为True，则将图形保存到tensorboard日志文件夹中
        device {torch.device}     -- 是否是GPU

    Returns:
        writer -- tensorboard SummaryWriter object
    """
    # 建立一个tensorboard SummaryWriter，并传入文件路径
    writer = SummaryWriter(log_dir=log_dir)
    # 判断是否存储图结构
    if save_graph:
        # 从迭代器中读取一个batch的images和labels
        images, labels = next(iter(train_loader))
        # 将数据装载到GPU上
        images = images.to(device)
        try:
            # 将图表数据添加到writer中
            writer.add_graph(model, images)
        except Exception as e:
            # 如果存储失败的话，报错！
            print("Failed to save model graph: {}".format(e))
    return writer


def train(model, optimizer, loss_fn, train_loader, val_loader,
          log_dir, device, epochs, log_interval,
          load_weight_path=None, save_graph=True):
    """训练逻辑

    Arguments:
        model {pytorch model}       -- 需要训练的模型
        optimizer {torch optim}     -- 优化器
        loss_fn                     -- 损失函数
        train_loader {dataloader}   -- 训练集加载器
        val_loader {dataloader}     -- 验证集加载器
        log_dir {str}               -- 日志文件夹
        device {torch.device}       -- 计算设备
        epochs {int}                -- 轮数
        log_interval {int}          -- 每次训练经过多少次间隔进行日志保留

    Keyword Arguments:
        load_weight_path {str} -- 要加载的模型权重路径（默认值：{None}）
        save_graph {bool}      -- 是否保存模型图（默认值：{True}）

    Returns:
        None
    """
    # 将模型装载到GPU上
    model.to(device)

    # 如果load_weight_path存在，则载入该权重到模型中
    if load_weight_path is not None:
        model.load_state_dict(torch.load(load_weight_path))

    # 将模型参数传入优化器(Adam)中
    optimizer = optimizer(model.parameters())

    # 训练过程
    def process_function(engine, batch):
        # 模型开启训练模式
        model.train()
        # 优化器梯度归零
        optimizer.zero_grad()
        # 获取一个batch中的数据
        x, _ = batch
        # 将数据载入GPU
        x = x.to(device)
        # 将数据导入模型
        y = model(x)
        # 计算损失
        loss = loss_fn(y, x)
        # 反向传播，得到每个参数的梯度值
        loss.backward()
        # 梯度下降，执行一步参数更新
        optimizer.step()
        # 返回loss的浮点数类型
        return loss.item()

    # 评估过程
    def evaluate_function(engine, batch):
        # 模型开启验证模式
        model.eval()
        with torch.no_grad():
            # 获取一个batch中的数据
            x, _ = batch
            # 将数据载入GPU
            x = x.to(device)
            # 将数据导入模型
            y = model(x)
            # 计算损失
            loss = loss_fn(y, x)
            # 返回loss的浮点数类型
            return loss.item()

    # 创建训练Engine和验证Engine
    trainer = Engine(process_function)
    evaluator = Engine(evaluate_function)

    # 计算指标的运行平均值
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x).attach(evaluator, 'loss')

    # 创建一个tensorboard数据
    writer = create_summary_writer(model, train_loader, log_dir, save_graph, device)

    def score_function(engine):
        return -engine.state.metrics['loss']

    to_save = {'model': model}

    # 创建一个检查点用来定期保存
    handler = Checkpoint(
        to_save,
        DiskSaver(os.path.join(log_dir, 'models'), create_dir=True),
        n_saved=None, filename_prefix='best', score_function=score_function,
        # filename_prefix='best', score_function=score_function,
        score_name="loss",
        global_step_transform=global_step_from_engine(trainer))

    # 添加在触发指定事件时要执行的事件处理程序
    evaluator.add_event_handler(Events.COMPLETED, handler)

    # 迭代结束时触发
    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        print(
            f"Epoch[{engine.state.epoch}] Iteration[{engine.state.iteration}/"
            f"{len(train_loader)}] Loss: {engine.state.output:.12f}")
        writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    # 每轮结束时触发
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_loss = metrics["loss"]
        print(f"Training Results - Epoch: {engine.state.epoch} Avg loss: {avg_loss:.12f}")
        writer.add_scalar("training/avg_loss", avg_loss, engine.state.epoch)

    # 每轮结束时触发
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_loss = metrics["loss"]
        print(f"Validation Results - Epoch: {engine.state.epoch} Avg loss: {avg_loss:.12f}")
        writer.add_scalar("validation/avg_loss", avg_loss, engine.state.epoch)

    # 运行trainer
    trainer.run(train_loader, max_epochs=epochs)

    # 关闭writer
    writer.close()
