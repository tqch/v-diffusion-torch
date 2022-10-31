import os
import json
import torch
from datetime import datetime
from torch.optim import AdamW, lr_scheduler
from v_diffusion import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing import errors
from functools import partial


@errors.record
def main(args):

    distributed = args.distributed

    def logger(msg, **kwargs):
        if not distributed or dist.get_rank() == 0:
            print(msg, **kwargs)

    root = os.path.expanduser(args.root)
    dataset = args.dataset

    in_channels = DATA_INFO[dataset]["channels"]
    image_res = DATA_INFO[dataset]["resolution"]
    image_shape = (in_channels, ) + image_res

    multitags = DATA_INFO[dataset].get("multitags", False)
    if args.use_cfg:
        num_classes = DATA_INFO[dataset].get("num_classes", 0)
        w_guide = args.w_guide
        p_uncond = args.p_uncond
    else:
        num_classes = 0
        w_guide = 0.
        p_uncond = 0.

    # set seed for all rngs
    seed = args.seed
    seed_all(seed)

    configs_path = os.path.join(args.config_dir, args.dataset + ".json")
    with open(configs_path, "r") as f:
        configs: dict = json.load(f)

    # train parameters
    gettr = partial(get_param, configs_1=configs.get("train", {}), configs_2=args)
    batch_size = gettr("batch_size")
    beta1, beta2 = gettr("beta1"), gettr("beta2")
    weight_decay = gettr("weight_decay")
    lr = gettr("lr")
    epochs = gettr("epochs")
    grad_norm = gettr("grad_norm")
    warmup = gettr("warmup")
    train_device = torch.device(args.train_device)
    eval_device = torch.device(args.eval_device)

    # diffusion parameters
    getdif = partial(get_param, configs_1=configs.get("diffusion", {}), configs_2=args)
    logsnr_schedule = getdif("logsnr_schedule")
    logsnr_min, logsnr_max = getdif("logsnr_min"), getdif("logsnr_max")
    train_timesteps = getdif("train_timesteps")
    sample_timesteps = getdif("sample_timesteps")
    reweight_type = getdif("reweight_type")
    logsnr_fn = get_logsnr_schedule(logsnr_schedule, logsnr_min=logsnr_min, logsnr_max=logsnr_max)
    model_out_type = getdif("model_out_type")
    model_var_type = getdif("model_var_type")
    loss_type = getdif("loss_type")

    diffusion = GaussianDiffusion(
        logsnr_fn=logsnr_fn,
        sample_timesteps=sample_timesteps,
        model_out_type=model_out_type,
        model_var_type=model_var_type,
        reweight_type=reweight_type,
        loss_type=loss_type,
        intp_frac=args.intp_frac,
        w_guide=w_guide,
        p_uncond=p_uncond
    )

    # denoise parameters
    # currently, model_var_type = "learned" is not supported
    # out_channels = 2 * in_channels if model_var_type == "learned" else in_channels
    out_channels = 2 * in_channels if model_out_type == "both" else in_channels
    _model = UNet(
        out_channels=out_channels,
        num_classes=num_classes,
        multitags=multitags,
        **configs["denoise"])

    if distributed:
        # check whether torch.distributed is available
        # CUDA devices are required to run with NCCL backend
        assert dist.is_available() and torch.cuda.is_available()
        dist.init_process_group("nccl")
        rank = dist.get_rank()  # global process id across all node(s)
        local_rank = int(os.environ["LOCAL_RANK"])  # local device id on a single node
        _model = _model.to(rank)
        model = DDP(_model, device_ids=[local_rank, ])
        train_device = torch.device(f"cuda:{local_rank}")
    else:
        rank = local_rank = 0  # main process by default
        model = _model.to(train_device)

    optimizer = AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    # Note1: lr_lambda is used to calculate the **multiplicative factor**
    # Note2: index starts at 0
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min((t + 1) / warmup, 1.0)) if warmup > 0 else None

    split = "all" if dataset == "celeba" else "train"
    num_workers = args.num_workers
    trainloader, sampler = get_dataloader(
        dataset, batch_size=batch_size, split=split, val_size=0., random_seed=seed,
        root=root, drop_last=True, pin_memory=True, num_workers=num_workers, distributed=distributed
    )  # drop_last to have a static input shape; num_workers > 0 to enable asynchronous data loading

    configs["train"]["epochs"] = epochs
    configs["use_ema"] = args.use_ema
    configs["conditional"] = {
        "use_cfg": args.use_cfg,
        "w_guide": w_guide,
        "p_uncond": p_uncond
    }
    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S%f")

    chkpt_dir = args.chkpt_dir
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)

    # keep a record of hyperparameter settings used for current experiment
    with open(os.path.join(chkpt_dir, f"exp_{timestamp}.json"), "w") as f:
        json.dump(configs, f)

    chkpt_path = os.path.join(chkpt_dir, args.chkpt_name or f"vdpm_{dataset}.pt")
    chkpt_intv = args.chkpt_intv
    logger(f"Checkpoint will be saved to {os.path.abspath(chkpt_path)}", end=" ")
    logger(f"every {chkpt_intv} epoch(s)")

    image_dir = os.path.join(args.image_dir, f"{dataset}")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    image_intv = args.image_intv
    num_save_images = args.num_save_images
    logger(f"Generated images (x{num_save_images}) will be saved to {os.path.abspath(image_dir)}", end=" ")
    logger(f"every {image_intv} epoch(s)")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        diffusion=diffusion,
        timesteps=train_timesteps,
        epochs=epochs,
        trainloader=trainloader,
        sampler=sampler,
        scheduler=scheduler,
        use_cfg=args.use_cfg,
        use_ema=args.use_ema,
        grad_norm=grad_norm,
        shape=image_shape,
        device=train_device,
        chkpt_intv=chkpt_intv,
        image_intv=image_intv,
        num_save_images=num_save_images,
        ema_decay=args.ema_decay,
        rank=rank,
        distributed=distributed
    )
    evaluator = Evaluator(dataset=dataset, device=eval_device) if args.eval else None
    # in case of elastic launch, resume should always be turned on
    resume = args.resume or distributed
    if resume:
        try:
            map_location = {"cuda:0": f"cuda:{local_rank}"} if distributed else train_device
            trainer.load_checkpoint(chkpt_path, map_location=map_location)
        except FileNotFoundError:
            logger("Checkpoint file does not exist!")
            logger("Starting from scratch...")

    # use cudnn benchmarking algorithm to select the best conv algorithm
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
        logger(f"cuDNN benchmark: ON")

    logger("Training starts...", flush=True)
    trainer.train(
        evaluator,
        chkpt_path=chkpt_path,
        image_dir=image_dir,
        use_ddim=args.use_ddim)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "celeba"], default="cifar10")
    parser.add_argument("--root", default="~/datasets", type=str, help="root directory of datasets")
    parser.add_argument("--epochs", default=120, type=int, help="total number of training epochs")
    parser.add_argument("--lr", default=0.0002, type=float, help="learning rate")
    parser.add_argument("--beta1", default=0.9, type=float, help="beta_1 in Adam")
    parser.add_argument("--beta2", default=0.999, type=float, help="beta_2 in Adam")
    parser.add_argument("--weight-decay", default=0., type=float,
                        help="decoupled weight_decay factor in Adam")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--train-timesteps", default=0, type=int, help=(
        "number of diffusion steps for training (0 indicates continuous training)"))
    parser.add_argument("--sample-timesteps", default=256, type=int, help="number of diffusion steps for sampling")
    parser.add_argument("--logsnr-schedule", choices=["linear", "sigmoid", "cosine", "legacy"], default="cosine")
    parser.add_argument("--logsnr-max", default=20., type=float)
    parser.add_argument("--logsnr-min", default=-20., type=float)
    parser.add_argument("--model-out-type", choices=["x_0", "eps", "both", "v"], default="both", type=str)
    parser.add_argument("--model-var-type", choices=["fixed_small", "fixed_large", "fixed_medium"], default="fixed_large", type=str)
    parser.add_argument("--reweight-type", choices=["constant", "snr", "truncated_snr", "alpha2"], default="truncated_snr", type=str)
    parser.add_argument("--loss-type", choices=["kl", "mse"], default="mse", type=str)
    parser.add_argument("--intp-frac", default=0., type=float)
    parser.add_argument("--use-cfg", action="store_true", help="whether to use classifier-free guidance")
    parser.add_argument("--w-guide", default=0.1, type=float, help="classifier-free guidance strength")
    parser.add_argument("--p-uncond", default=0.1, type=float, help="probability of unconditional training")
    parser.add_argument("--num-workers", default=4, type=int, help="number of workers for data loading")
    parser.add_argument("--train-device", default="cuda:0", type=str)
    parser.add_argument("--eval-device", default="cuda:0", type=str)
    parser.add_argument("--image-dir", default="./images/train", type=str)
    parser.add_argument("--image-intv", default=10, type=int)
    parser.add_argument("--num-save-images", default=64, type=int, help="number of images to generate & save")
    parser.add_argument("--config-dir", default="./configs", type=str)
    parser.add_argument("--chkpt-dir", default="./chkpts", type=str)
    parser.add_argument("--chkpt-name", default="", type=str)
    parser.add_argument("--chkpt-intv", default=120, type=int, help="frequency of saving a checkpoint")
    parser.add_argument("--seed", default=1234, type=int, help="random seed")
    parser.add_argument("--resume", action="store_true", help="to resume training from a checkpoint")
    parser.add_argument("--eval", action="store_true", help="whether to evaluate fid during training")
    parser.add_argument("--use-ema", action="store_true", help="whether to use exponential moving average")
    parser.add_argument("--use-ddim", action="store_true", help="whether to use DDIM sampler")
    parser.add_argument("--ema-decay", default=0.9999, type=float, help="decay factor of ema")
    parser.add_argument("--distributed", action="store_true", help="whether to use distributed training")

    main(parser.parse_args())
