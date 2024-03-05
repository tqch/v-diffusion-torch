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

    config_path = args.config_path
    exp_name = args.exp_name or os.path.splitext(os.path.basename(config_path))[0]
    with open(config_path, "r") as f:
        config: dict = json.load(f)
    with open(args.default_config_path, "r") as f:
        defaults: dict = json.load(f)
    fill_with_defaults(config, defaults)

    # dataset parameters
    update_data = partial(update_config, old_config=config.get("data", {}), new_config=args)
    dataset = config["data"]["name"]
    root = update_data("root", "data_root")
    if "~" in root:
        root = os.path.expanduser(root)
    if "$" in root:
        root = os.path.expandvars(root)

    in_channels = DATA_INFO[dataset]["channels"]
    image_res = DATA_INFO[dataset]["resolution"]
    image_shape = (in_channels, ) + image_res

    # conditional parameters
    update_cond = partial(update_config, old_config=config.get("conditional", {}), new_config=args)
    use_cfg = update_cond("use_cfg", logical_op="OR")
    w_guide = update_cond("w_guide")
    p_uncond = update_cond("p_uncond")

    multitags = DATA_INFO[dataset].get("multitags", False)
    if use_cfg:
        num_classes = DATA_INFO[dataset].get("num_classes", 0)
    else:
        num_classes = 0

    # train parameters
    update_train = partial(update_config, old_config=config.get("train", {}), new_config=args)
    epochs = update_train("epochs")
    seed = update_train("seed")
    batch_size = update_train("batch_size")
    beta1, beta2 = update_train("beta1"), update_train("beta2")
    weight_decay = update_train("weight_decay")
    lr = update_train("lr")
    grad_norm = update_train("grad_norm")
    warmup = update_train("warmup")
    use_ema = update_train("use_ema", logical_op="OR")
    ema_decay = update_train("ema_decay")
    ckpt_intv = update_train("ckpt_intv")
    image_intv = update_train("image_intv")
    num_save_images = update_train("num_save_images")
    max_ckpts_kept = update_train("max_ckpts_kept")
    save_rng_state = update_train("save_rng_state", logical_op="OR")

    # set seed for all rngs
    seed_all(seed)

    train_device = torch.device(args.train_device)
    eval_device = torch.device(args.eval_device)

    # diffusion parameters
    update_diff = partial(update_config, old_config=config.get("diffusion", {}), new_config=args)
    logsnr_schedule = update_diff("logsnr_schedule")
    logsnr_min, logsnr_max = update_diff("logsnr_min"), update_diff("logsnr_max")
    train_timesteps = update_diff("train_timesteps")
    sample_timesteps = update_diff("sample_timesteps")
    reweight_type = update_diff("reweight_type")
    model_out_type = update_diff("model_out_type")
    model_var_type = update_diff("model_var_type")
    intp_frac = update_diff("intp_frac")
    loss_type = update_diff("loss_type")
    allow_rescale = update_diff("allow_rescale", logical_op="OR")
    x0eps_coef = update_diff("x0eps_coef", logical_op="OR")

    t_rescale = (train_timesteps == 0) and allow_rescale
    logsnr_fn = get_logsnr_schedule(logsnr_schedule, logsnr_min=logsnr_min, logsnr_max=logsnr_max, rescale=t_rescale)

    diffusion = GaussianDiffusion(
        logsnr_fn=logsnr_fn,
        sample_timesteps=sample_timesteps,
        model_out_type=model_out_type,
        model_var_type=model_var_type,
        reweight_type=reweight_type,
        loss_type=loss_type,
        intp_frac=intp_frac,
        w_guide=w_guide,
        p_uncond=p_uncond,
        x0eps_coef=x0eps_coef,
    )

    # model parameters
    update_model = partial(update_config, old_config=config.get("model", {}), new_config=args)
    update_model("use_xformers", logical_op="OR")

    # currently, model_var_type = "learned" is not supported
    # out_channels = 2 * in_channels if model_var_type == "learned" else in_channels
    if "in_channels" in config["model"]:
        assert config["model"]["in_channels"] == in_channels
    else:
        config["model"]["in_channels"] = in_channels
    if "out_channels" not in config["model"]:
        assert "model_out_type" in config["diffusion"]
        out_channels = 2 * in_channels if model_out_type == "both" else in_channels
        config["model"]["out_channels"] = out_channels
    _model = UNet(
        num_classes=num_classes,
        multitags=multitags,
        **config["model"])

    if distributed:
        # check whether torch.distributed is available
        # CUDA devices are required to run with NCCL backend
        assert dist.is_available() and torch.cuda.is_available()
        dist.init_process_group("nccl")
        rank = dist.get_rank()  # global process id across all node(s)
        world_size = dist.get_world_size()  # total number of processes
        local_rank = int(os.environ["LOCAL_RANK"])  # local device id on a single node
        torch.cuda.set_device(local_rank)
        _model.cuda()
        model = DDP(_model, device_ids=[local_rank, ])
        train_device = eval_device = torch.device(f"cuda:{local_rank}")
    else:
        rank = local_rank = 0  # main process by default
        world_size = 1
        model = _model.to(train_device)
    is_leader = rank == 0

    optimizer = AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    # Note1: lr_lambda is used to calculate the **multiplicative factor**
    # Note2: index starts at 0
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min((t + 1) / warmup, 1.0)) if warmup > 0 else None

    split = "all" if dataset == "celeba" else "train"
    num_workers = args.num_workers
    trainloader, sampler = get_dataloader(
        dataset, batch_size=batch_size // args.num_accum, split=split, val_size=0., random_seed=seed,
        root=root, drop_last=True, pin_memory=True, num_workers=num_workers, distributed=distributed,
        is_leader=is_leader
    )  # drop_last to have a static input shape; num_workers > 0 to enable asynchronous data loading

    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S%f")

    exp_dir = os.path.join(args.exp_dir, f"dpm_{exp_name}", timestamp)

    ckpt_dir = os.path.join(exp_dir, "ckpts")
    ckpt_path = os.path.join(ckpt_dir, "ckpt_{epoch}.pt")
    logger(f"Checkpoint will be saved to {os.path.abspath(ckpt_dir)}", end=" ")
    logger(f"every {ckpt_intv} epoch(s)")

    image_dir = os.path.join(exp_dir, "images")
    if is_leader and not os.path.exists(image_dir):
        os.makedirs(image_dir)
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
        use_cfg=use_cfg,
        use_ema=use_ema,
        grad_norm=grad_norm,
        num_accum=args.num_accum,
        shape=image_shape,
        device=train_device,
        ckpt_intv=ckpt_intv,
        max_ckpts_kept=max_ckpts_kept,
        image_intv=image_intv,
        num_save_images=num_save_images,
        eval_intv=args.eval_intv,
        ema_decay=ema_decay,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        save_rng_state=save_rng_state,
    )
    evaluator = Evaluator(dataset=dataset, device=eval_device) if args.eval else None
    # in case of elastic launch, resume should always be turned on
    resume = args.resume or distributed
    if resume:
        try:
            map_location = {"cuda:0": f"cuda:{local_rank}"} if distributed else train_device
            from_ckpt = args.from_ckpt or ckpt_path
            trainer.load_checkpoint(from_ckpt, map_location=map_location)
            logger(f"Successfully loaded checkpoint from {os.path.abspath(from_ckpt)}!")
        except FileNotFoundError:
            logger("Checkpoint file does not exist!")
            logger("Starting from scratch...")

    # speedup parameters
    update_speedup = partial(update_config, old_config=config.get("speedup", {}), new_config=args)
    cudnn_benchmark = update_speedup("cudnn_benchmark", logical_op="OR")
    allow_tf32 = update_speedup("allow_tf32", logical_op="OR")
    allow_fp16 = update_speedup("allow_fp16", logical_op="OR")
    allow_bf16 = update_speedup("allow_bf16", logical_op="OR")

    allow_tf32 = any(
        f"NVIDIA {x}" in torch.cuda.get_device_name()
        for x in ("A", "H", "RTX A", "RTX 30", "RTX 40", "RTX 50", "RTX 60")
    ) and allow_tf32

    if torch.backends.cudnn.is_available():
        # use cudnn benchmarking algorithm to select the best conv algorithm
        torch.backends.cudnn.benchmark = cudnn_benchmark
        logger(f"cuDNN benchmark: {'ON' if cudnn_benchmark else 'OFF'}")
        # TF32 tensor cores are designed to achieve better performance on matmul and convolutions on torch.float32
        # tensors by rounding input data to have 10 bits of mantissa, and accumulating results with FP32 precision,
        # maintaining FP32 dynamic range.
        # source: https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere
        # On Ampere and later CUDA devices, matrix multiplications and convolutions
        # can use the TensorFloat-32 (TF32) mode for faster, but slightly less accurate computations.
        # source: https://huggingface.co/docs/diffusers/en/optimization/fp16
        torch.backends.cudnn.allow_tf32 |= allow_tf32  # default to True; disabling will slow down training
        logger(f"TF32 conv: {'ON' if torch.backends.cudnn.allow_tf32 else 'OFF'}")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        logger(f"TF32 matmul: {'ON' if torch.backends.cuda.matmul.allow_tf32 else 'OFF'}")
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = allow_fp16
        logger(f"{'Enabled' if allow_fp16 else 'Disabled'} reduced precision reductions in fp16 GEMMs")
        if torch.version.__version__.split("+")[0].split(".") >= ["2", "0", "0"]:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = allow_bf16
            logger(f"{'Enabled' if allow_fp16 else 'Disabled'} reduced precision reductions in bf16 GEMMs")

    if is_leader:
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        # keep a record of hyperparameter settings used for current experiment
        with open(os.path.join(exp_dir, f"config.json"), "w") as f:
            config["args"] = vars(args)
            json.dump(config, f, indent=2)

    logger("Training starts...", flush=True)
    trainer.train(
        evaluator,
        ckpt_path=ckpt_path,
        image_dir=image_dir,
        use_ddim=args.use_ddim,
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str, help="root directory of datasets")
    parser.add_argument("--epochs", type=int, help="total number of training epochs")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--beta1", type=float, help="beta_1 in Adam")
    parser.add_argument("--beta2", type=float, help="beta_2 in Adam")
    parser.add_argument("--weight-decay", type=float, help="decoupled weight_decay factor in Adam")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-accum", type=int, default=1, help="number of batches before weight update, a.k.a. gradient accumulation")
    parser.add_argument("--train-timesteps", type=int, help="number of diffusion steps for training (0 indicates continuous training)")
    parser.add_argument("--sample-timesteps", type=int, help="number of diffusion steps for sampling")
    parser.add_argument("--logsnr-schedule", choices=["linear", "sigmoid", "cosine", "legacy"])
    parser.add_argument("--logsnr-max", type=float)
    parser.add_argument("--logsnr-min", type=float)
    parser.add_argument("--model-out-type", type=str, choices=["x_0", "eps", "both", "v"])
    parser.add_argument("--model-var-type", type=str, choices=["fixed_small", "fixed_large", "fixed_medium"])
    parser.add_argument("--reweight-type", type=str, choices=["constant", "snr", "snr_trunc", "snr_1plus"])
    parser.add_argument("--loss-type", type=str, choices=["kl", "mse"])
    parser.add_argument("--intp-frac", type=float)
    parser.add_argument("--w-guide", type=float, help="classifier-free guidance strength")
    parser.add_argument("--p-uncond", type=float, help="probability of unconditional training")
    parser.add_argument("--num-workers", type=int, default=4, help="number of workers for data loading")
    parser.add_argument("--train-device", type=str, default="cuda:0")
    parser.add_argument("--eval-device", type=str, default="cuda:0")
    parser.add_argument("--image-intv", type=int)
    parser.add_argument("--num-save-images", type=int, help="number of images to generate & save")
    parser.add_argument("--use-ddim", action="store_true", help="whether to use DDIM sampler")
    parser.add_argument("--config-path", required=True, type=str)
    parser.add_argument("--default-config-path", default="./configs/defaults.json", type=str)
    parser.add_argument("--exp-dir", type=str, default="./exps")
    parser.add_argument("--exp-name", type=str)
    parser.add_argument("--ckpt-intv", type=int, help="frequency of saving a checkpoint")
    parser.add_argument("--save-rng-state", action="store_true", help="whether to save the rng state of each device")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--resume", action="store_true", help="to resume training from a checkpoint")
    parser.add_argument("--from-ckpt", type=str, help="from which checkpoint to resume")
    parser.add_argument("--eval", action="store_true", help="whether to evaluate fid during training")
    parser.add_argument("--eval-intv", type=int, default=128, help="frequency of evaluating the model")
    parser.add_argument("--ema-decay", type=float, help="decay factor of ema")
    parser.add_argument("--distributed", action="store_true", help="whether to use distributed training")
    parser.add_argument("--cudnn-benchmark", action="store_true", help="whether to enable cuDNN benchmark")
    parser.add_argument("--allow-tf32", action="store_true", help="whether allowing using TensorFloat32 (TF32)")
    parser.add_argument("--allow-fp16", action="store_true", help="whether allowing using float16 (fp16)")
    parser.add_argument("--allow-bf16", action="store_true", help="whether allowing using bfloat16 (bf16)")
    parser.add_argument("--use-xformers", action="store_true", help="whether to use memory efficient attention")
    parser.add_argument("--max-ckpts-kept", type=int, help="maximum number of checkpoints to keep on disk (none for no cap)")

    # "OR"-type flags: use_cfg, use_ema, allow_rescale, x0eps_coef
    parser.add_argument("--use-cfg", action="store_true", help="whether to use classifier-free guidance")
    parser.add_argument("--use-ema", action="store_true", help="whether to use exponential moving average")

    # set the following flags to replicate google-research implementation
    # reference: https://github.com/google-research/google-research/blob/master/ddpm_w_distillation/ddpm_w_distillation/dpm.py
    parser.add_argument("--allow-rescale", action="store_true", help="whether to allow in-place adjustment of t")
    parser.add_argument("--x0eps-coef", action="store_true", help="whether the posterior mean should be expressed in terms of x0 and eps")

    main(parser.parse_args())
