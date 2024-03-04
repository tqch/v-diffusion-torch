import glob
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.distributed as tdist
from torch.nn.parallel import DistributedDataParallel
from .utils import save_image, EMA
from .metrics.fid_score import InceptionStatistics, get_precomputed, calc_fd
from tqdm import tqdm
from contextlib import nullcontext
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


class DummyScheduler:
    @staticmethod
    def step():
        pass

    def load_state_dict(self, state_dict):
        pass

    @staticmethod
    def state_dict():
        return None


class RunningStatistics:
    def __init__(self, **kwargs):
        self.count = 0
        self.stats = []
        for k, v in kwargs.items():
            self.stats.append((k, v or 0))
        self.stats = dict(self.stats)

    def reset(self):
        self.count = 0
        for k in self.stats:
            self.stats[k] = 0

    def update(self, n, **kwargs):
        self.count += n
        for k, v in kwargs.items():
            self.stats[k] = self.stats.get(k, 0) + v

    def extract(self):
        avg_stats = []
        for k, v in self.stats.items():
            avg_stats.append((k, v / self.count))
        return dict(avg_stats)

    def __repr__(self):
        out_str = "Count(s): {}\n"
        out_str += "Statistics:\n"
        for k in self.stats:
            out_str += f"\t{k} = {{{k}}}\n"  # double curly-bracket to escape
        return out_str.format(self.count, **self.stats)


class Trainer:
    def __init__(
            self,
            model,
            optimizer,
            diffusion,
            timesteps,
            epochs,
            trainloader,
            sampler=None,
            scheduler=None,
            use_cfg=False,
            use_ema=False,
            grad_norm=1.0,
            num_accum=1,
            shape=None,
            device=torch.device("cpu"),
            ckpt_intv=512,  # save a checkpoint every <ckpt_intv> epochs
            max_ckpts_kept=-1,  # maximum number of checkpoints to keep
            image_intv=64,  # generate images every <image_intv> epochs
            eval_intv=128,  # evaluate images every <eval_intv> epochs
            num_save_images=64,
            ema_decay=0.9999,
            distributed=False,
            rank=0,  # process id for distributed training
            world_size=1,  # total number of processes
    ):
        self.model = model
        self.optimizer = optimizer
        self.diffusion = diffusion
        self.timesteps = timesteps
        self.epochs = epochs
        self.start_epoch = 0
        self.trainloader = trainloader
        self.sampler = sampler
        if shape is None:
            shape = next(iter(trainloader))[0].shape[1:]
        self.shape = shape
        self.scheduler = DummyScheduler() if scheduler is None else scheduler

        self.grad_norm = grad_norm
        self.num_accum = num_accum
        self.device = device
        self.ckpt_intv = ckpt_intv
        self.max_ckpts_kept = max_ckpts_kept
        self.image_intv = image_intv
        self.eval_intv = eval_intv
        self.num_save_images = num_save_images

        if distributed:
            assert sampler is not None
        self.distributed = distributed
        self.rank = rank
        self.is_leader = rank == 0
        self.world_size = world_size

        assert num_save_images % world_size == 0
        self.local_num_save_images = num_save_images // world_size

        # maintain a process-specific generator
        self.generator = torch.Generator(device).manual_seed(8191 + rank)
        self.label_seed = 127 + rank  # process-specific seed
        self.sample_seed = 131071 + rank  # process-specific seed

        self.use_cfg = use_cfg
        self.use_ema = use_ema
        if self.is_leader and use_ema:
            self.ema = EMA(self.model, decay=ema_decay)
        else:
            self.ema = nullcontext()

        self.stats = RunningStatistics(loss=None)

    def loss(self, x, y):
        B = x.shape[0]
        T = self.timesteps
        if T > 0:
            t = torch.randint(
                T, size=(B,), dtype=torch.float64, device=self.device, generator=self.generator
            ).add(1).div(self.timesteps)
        else:
            t = torch.rand((B,), dtype=torch.float64, device=self.device, generator=self.generator)
        noise = torch.empty_like(x).normal_(generator=self.generator)
        loss = self.diffusion.train_loss(self.model, x_0=x, t=t, y=y, noise=noise)
        assert loss.shape == (B,)
        return loss

    def step(self, x, y, update=True):
        B = x.shape[0]
        loss = self.loss(x, y).mean()
        loss.div(self.num_accum).backward()
        loss = loss.detach()
        if self.distributed:
            tdist.reduce(loss, dst=0, op=tdist.ReduceOp.SUM)  # synchronize losses
            loss.div_(self.world_size)
        if update:
            # gradient clipping by global norm
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            # adjust learning rate every step (warming up)
            self.scheduler.step()
            if self.is_leader and self.use_ema:
                assert isinstance(self.ema, EMA)
                self.ema.update()
        self.stats.update(B, loss=loss.item() * B)

    def sample_fn(self, label, diffusion=None, use_ddim=False):
        if diffusion is None:
            diffusion = self.diffusion
        shape = (self.local_num_save_images, *self.shape)
        with self.ema:
            sample = diffusion.p_sample(
                denoise_fn=self.model, shape=shape, device=self.device, noise=None,
                label=label, seed=self.sample_seed, use_ddim=use_ddim)
        if self.distributed:
            # equalizes GPU memory usages across all processes within the same process group
            sample_list = [torch.zeros(shape, device=self.device) for _ in range(self.world_size)]
            dist.all_gather(sample_list, sample.to(self.device))
            sample = torch.cat(sample_list, dim=0).cpu()
        assert sample.grad is None
        return sample

    def sample_labels(self):
        if self.multitags:
            inds = torch.randint(
                len(self.trainloader.dataset),
                size=(self.num_save_images,),
                generator=torch.Generator().manual_seed(self.label_seed)
            )
            labels = torch.as_tensor(
                self.trainloader.dataset.targets[inds], dtype=torch.float32)
        else:
            labels = torch.arange(self.num_classes, dtype=torch.float32) + 1
            repeats = torch.as_tensor([
                (self.num_save_images // self.num_classes
                 + int(i < self.num_save_images % self.num_classes))
                for i in range(self.num_classes)])
            labels = labels.repeat_interleave(repeats)
        return labels[self.rank * self.local_num_save_images: (self.rank + 1) * self.local_num_save_images]

    @property
    def num_classes(self):
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module.num_classes
        else:
            return self.model.num_classes

    @property
    def multitags(self):
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module.multitags
        else:
            return self.model.multitags

    def train(
            self,
            evaluator=None,
            ckpt_path=None,
            image_dir=None,
            use_ddim=False,
    ):

        nrow = 8
        labels = None
        if self.num_save_images:
            if self.num_classes:
                labels = self.sample_labels()
            if self.num_classes:
                nrow = math.ceil(self.num_save_images / self.num_classes)
            else:
                nrow = math.floor(math.sqrt(self.num_save_images))

        total_batches = 0
        for e in range(self.start_epoch, self.epochs):
            self.stats.reset()
            self.model.train()
            results = dict()
            if isinstance(self.sampler, DistributedSampler):
                self.sampler.set_epoch(e)
            with tqdm(
                    self.trainloader,
                    desc=f"{e + 1}/{self.epochs} epochs",
                    disable=not self.is_leader
            ) as t:
                for i, (x, y) in enumerate(t):
                    total_batches += 1
                    if not self.use_cfg:
                        y = None
                    self.step(
                        x.to(self.device),
                        y.float().to(self.device)
                        if y is not None else y,
                        update=total_batches % self.num_accum == 0
                    )
                    t.set_postfix(self.current_stats)

                if self.is_leader:
                    if not (e + 1) % self.eval_intv and evaluator is not None:
                        self.model.eval()
                        eval_results = evaluator.eval(self.sample_fn)
                        results.update(self.current_stats)
                        results.update(eval_results)
                        t.set_postfix(results)

            if not (e + 1) % self.image_intv and self.num_save_images and image_dir:
                self.model.eval()
                x = self.sample_fn(labels, use_ddim=use_ddim)
                if self.is_leader:
                    save_image(x, os.path.join(image_dir, f"{e + 1}.png"), nrow=nrow)
            if self.is_leader:
                if not (e + 1) % self.ckpt_intv and ckpt_path and self.max_ckpts_kept != 0:
                    if not results:
                        results = self.current_stats
                    self.save_checkpoint(ckpt_path, epoch=e + 1, **results)
            if self.distributed:
                dist.barrier()  # synchronize all processes here

    @property
    def trainees(self):
        roster = ["model", "optimizer"]
        if self.use_ema:
            roster.append("ema")
        if self.scheduler is not None:
            roster.append("scheduler")
        return roster

    @property
    def current_stats(self):
        return self.stats.extract()

    def load_checkpoint(self, ckpt_path, map_location):
        ckpt = torch.load(ckpt_path, map_location=map_location)
        for trainee in self.trainees:
            try:
                getattr(self, trainee).load_state_dict(ckpt[trainee])
            except RuntimeError:
                _ckpt = ckpt[trainee]["shadow"] if trainee == "ema" else ckpt[trainee]
                for k in list(_ckpt.keys()):
                    if k.split(".")[0] == "module":
                        _ckpt[".".join(k.split(".")[1:])] = _ckpt.pop(k)
                getattr(self, trainee).load_state_dict(ckpt[trainee])
            except AttributeError:
                continue
        self.start_epoch = ckpt["epoch"]

    def save_checkpoint(self, ckpt_path, **extra_info):
        ckpt = []
        for k, v in self.named_state_dicts():
            ckpt.append((k, v))
        for k, v in extra_info.items():
            ckpt.append((k, v))
        if "epoch" in extra_info:
            ckpt_path = ckpt_path.format(epoch=extra_info["epoch"])
        else:
            ckpt_path = ckpt_path.format(epoch="latest")
        torch.save(dict(ckpt), ckpt_path)
        ckpts = glob.glob(f"{os.path.dirname(ckpt_path)}/*.pt")
        if self.max_ckpts_kept != -1 and len(ckpts) > self.max_ckpts_kept:
            oldest_ckpt_path = ckpts[np.argmin(list(map(os.path.getctime, ckpts)))]
            os.remove(oldest_ckpt_path)

    def named_state_dicts(self):
        for k in self.trainees:
            yield k, getattr(self, k).state_dict()


class Evaluator:
    def __init__(
            self,
            dataset,
            diffusion=None,
            eval_batch_size=256,
            max_eval_count=10000,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.diffusion = diffusion
        # inception stats
        self.istats = InceptionStatistics(device=device)
        self.eval_batch_size = eval_batch_size
        self.max_eval_count = max_eval_count
        self.device = device
        self.target_mean, self.target_var = get_precomputed(dataset)

    def eval(self, sample_fn):
        self.istats.reset()
        for _ in range(0, self.max_eval_count + self.eval_batch_size, self.eval_batch_size):
            x = sample_fn(self.eval_batch_size, diffusion=self.diffusion)
            self.istats(x.to(self.device))
        gen_mean, gen_var = self.istats.get_statistics()
        return {"fid": calc_fd(gen_mean, gen_var, self.target_mean, self.target_var)}
