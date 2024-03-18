if __name__ == "__main__":
    import os
    import json
    import math
    import uuid
    import torch
    from datetime import datetime
    from tqdm import trange
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor
    from v_diffusion import *
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--data-root", type=str, default="~/datasets")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--total-size", type=int, default=50000)
    parser.add_argument("--default-config-path", default="./configs/defaults.json", type=str)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="./images/eval")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--use-ddim", action="store_true")
    parser.add_argument("--sample-timesteps", type=int, default=1024)
    parser.add_argument("--uncond", action="store_true")
    parser.add_argument("--w-guide", type=float, default=0.1)

    args = parser.parse_args()

    device = torch.device(args.device)

    ckpt_path = args.ckpt_path
    use_ema = args.use_ema
    if use_ema:
        state_dict = torch.load(ckpt_path, map_location=device)["ema"]["shadow"]
    else:
        state_dict = torch.load(ckpt_path, map_location=device)["model"]

    for k in list(state_dict.keys()):
        if k.startswith("module."):  # state_dict of DDP
            state_dict[k.split(".", maxsplit=1)[1]] = state_dict.pop(k)

    use_cfg = "class_embed" in {k.split(".")[0] for k in state_dict.keys()}

    config_path = args.config_path
    exp_name = os.path.splitext(os.path.basename(config_path))[0]  # truncated at file extension
    with open(config_path, "r") as f:
        config: dict = json.load(f)
    with open(args.default_config_path, "r") as f:
        defaults: dict = json.load(f)
    fill_with_defaults(config, defaults)
    dataset = config["data"]["name"]

    data_root = args.data_root
    if "~" in data_root:
        data_root = os.path.expanduser(data_root)
    if "$" in data_root:
        data_root = os.path.expandvars(data_root)
    in_channels = DATA_INFO[dataset]["channels"]
    image_res = DATA_INFO[dataset]["resolution"][0]
    multitags = DATA_INFO[dataset].get("multitags", False)
    if use_cfg:
        num_classes = DATA_INFO[dataset]["num_classes"]
        w_guide = 0. if args.uncond else args.w_guide
    else:
        num_classes = 0
        w_guide = 0

    diffusion_kwargs = config["diffusion"]
    logsnr_schedule = diffusion_kwargs.pop("logsnr_schedule")
    logsnr_max = diffusion_kwargs.pop("logsnr_max")
    logsnr_min = diffusion_kwargs.pop("logsnr_min")
    logsnr_fn = get_logsnr_schedule(
        logsnr_schedule, logsnr_min, logsnr_max, rescale=diffusion_kwargs.pop("allow_rescale"))
    diffusion_kwargs["sample_timesteps"] = args.sample_timesteps
    diffusion_kwargs.pop("train_timesteps")

    diffusion = GaussianDiffusion(
        logsnr_fn=logsnr_fn,
        w_guide=w_guide,
        **diffusion_kwargs)

    model_out_type = diffusion_kwargs.get("model_out_type", "both")
    out_channels = (2 if model_out_type == "both" else 1) * in_channels
    model = UNet(
        out_channels=out_channels,
        num_classes=num_classes,
        multitags=multitags,
        **config["model"],
    )
    model.to(device)

    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        if p.requires_grad:
            p.requires_grad_(False)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S%f")
    save_dir = os.path.join(args.save_dir, exp_name, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    batch_size = args.batch_size
    total_size = args.total_size
    num_eval_batches = math.ceil(total_size / batch_size)
    shape = (batch_size, 3, image_res, image_res)

    with open(os.path.join(save_dir, "args.txt"), "w") as f:
        json.dump(vars(args), f)

    def save_image(arr):
        with Image.fromarray(arr, mode="RGB") as im:
            im.save(f"{save_dir}/{uuid.uuid4()}.png")

    if torch.backends.cudnn.is_available():  # noqa
        torch.backends.cudnn.benchmark = True  # noqa

    uncond = args.uncond
    if multitags:
        labels = DATA_INFO[dataset]["data"](root=args.data_root, split="all").targets

        def get_label_loader(to_device):
            while True:
                if uncond:
                    yield torch.zeros((batch_size, num_classes), dtype=torch.float32, device=to_device)
                else:
                    yield labels[torch.randint(len(labels), size=(batch_size, ))].float().to(to_device)
    else:
        def get_label_loader(to_device):
            while True:
                if uncond:
                    yield torch.zeros((batch_size, ), dtype=torch.int64, device=to_device)
                else:
                    yield torch.randint(num_classes, size=(batch_size, ), device=to_device) + 1

    label_loader = get_label_loader(to_device=device)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        for i in trange(num_eval_batches):
            if i == num_eval_batches - 1:
                batch_size = total_size - i * batch_size
                shape = (batch_size, 3, image_res, image_res)

            x = diffusion.p_sample(
                model, shape=shape, device=device,
                noise=torch.randn(shape, device=device),
                label=next(label_loader)[:batch_size],
                use_ddim=args.use_ddim
            ).cpu()
            x = (x * 127.5 + 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
            pool.map(save_image, list(x))
