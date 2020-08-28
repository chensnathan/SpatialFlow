import os.path as osp

from mmcv.runner import Hook
from torch.utils.data import DataLoader


class EvalHook(Hook):
    """Evaluation hook.

    Note that if new arguments are added for EvalHook, tools/test.py may be
    effected.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        eval_kwargs (dict): Evaluation arguments of :func:`dataset.evaluate()`.
    """

    def __init__(self, dataloader, interval=1, cfg=None, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got'
                            f' {type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.eval_kwargs = eval_kwargs
        self.cfg = cfg

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        from mmdet.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results, cfg=self.cfg)

    def evaluate(self, runner, results, cfg=None):
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs, cfg=cfg)
        if isinstance(eval_res, dict):
            for name, val in eval_res.items():
                runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True


class DistEvalHook(EvalHook):
    """Distributed evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
    """

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 cfg=None,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got '
                            f'{type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.gpu_collect = gpu_collect
        self.eval_kwargs = eval_kwargs
        self.cfg = cfg

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        from mmdet.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results, cfg=self.cfg)
