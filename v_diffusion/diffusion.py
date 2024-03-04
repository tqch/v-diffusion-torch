import math
import numpy as np
import torch
import torch.nn.functional as F

try:
    from .functions import normal_kl, discretized_gaussian_loglik, flat_mean
except ImportError:
    import sys
    from pathlib import Path

    PROJ_DIR = str(Path(__file__).resolve().parents[1])
    if PROJ_DIR not in sys.path:
        sys.path.append(PROJ_DIR)
    from v_diffusion.functions import normal_kl, discretized_gaussian_loglik, flat_mean


def broadcast_to(
        arr, x,
        dtype=None, device=None, ndim=None):
    if x is not None:
        dtype = dtype or x.dtype
        device = device or x.device
        ndim = ndim or x.ndim
    out = torch.as_tensor(arr, dtype=dtype, device=device)
    return out.reshape((-1,) + (1,) * (ndim - 1))


def get_logsnr_schedule(
        schedule,
        logsnr_min: float = -20.,
        logsnr_max: float = 20.,
        rescale: bool = False,
):
    """
    schedule is named according to the relationship between alpha2 and t,
    i.e. alpha2 as a XX function of affine transformation of t (except for legacy)
    """

    logsnr_min, logsnr_max = torch.as_tensor(logsnr_min, dtype=torch.float64), \
        torch.as_tensor(logsnr_max, dtype=torch.float64)
    if schedule == "linear":
        def logsnr2t(logsnr):
            return torch.sigmoid(logsnr)

        def t2logsnr(t):
            return torch.logit(t)
    elif schedule == "sigmoid":
        logsnr_range = logsnr_max - logsnr_min

        def logsnr2t(logsnr):
            return (logsnr_max - logsnr) / logsnr_range

        def t2logsnr(t):
            return logsnr_max - t * logsnr_range
    elif schedule == "cosine":
        def logsnr2t(logsnr):
            return torch.atan(torch.exp(-0.5 * logsnr)).div(0.5 * torch.pi)

        def t2logsnr(t):
            return -2 * torch.log(torch.tan(t * torch.pi * 0.5))

        rescale = 2 / math.pi and rescale
    elif schedule == "legacy":
        """
        continuous version of the (discrete) linear schedule used by \
          Ho, Jonathan, Ajay Jain, and Pieter Abbeel. \
            "Denoising diffusion probabilistic models." \
              Advances in Neural Information Processing Systems 33 (2020): 6840-6851.
        """
        x_from = x_max = 0.9999
        x_min = 0.98
        slope = -0.0199

        def schedule_fn(t: torch.Tensor):
            x_to = broadcast_to(x_max, t).lerp(broadcast_to(x_min, t), t)
            log_alpha = 1000 / slope * (x_to * torch.log(x_to) - x_to - x_from * np.log(x_from) + x_from)
            return log_alpha - stable_log1mexp(log_alpha - 1e-9)

        return schedule_fn

    else:
        raise NotImplementedError

    t_from = logsnr2t(logsnr_max)
    t_to = logsnr2t(logsnr_min)

    def schedule_fn(t):
        _t = t.to(torch.float64)
        _t_from, _t_to = broadcast_to(t_from, _t), broadcast_to(t_to, _t)
        logsnr = t2logsnr(torch.lerp(_t_from, _t_to, _t))
        if rescale:
            if isinstance(rescale, bool):
                t.copy_(logsnr2t(logsnr).to(t.dtype))
            elif isinstance(rescale, float):
                t.mul_(rescale)
        return logsnr.to(t.dtype)

    return schedule_fn


def stable_log1mexp(x):
    """
    numerically stable version of log(1-exp(x)), x<0
    """
    assert torch.all(x < 0.)
    return torch.where(
        x < -9,
        torch.log1p(torch.exp(x).neg()),
        torch.log(torch.expm1(x).neg()))


def logsnr_to_posterior(
        logsnr_s, logsnr_t,
        var_type: str, intp_frac: float = None, x0eps_coef: bool = False
):
    # upcast to double precision to reduce precision loss
    logsnr_s, logsnr_t = (logsnr_s.to(torch.float64), logsnr_t.to(torch.float64))

    log_alpha_st = 0.5 * (F.logsigmoid(logsnr_s) - F.logsigmoid(logsnr_t))
    logr = logsnr_t - logsnr_s
    log_one_minus_r = stable_log1mexp(logr)
    # express posterior mean in terms of noise and data, o/w noisy data and data
    if x0eps_coef:
        # E[x_s|x_t] = mean_coef1 * eps + mean_coef2 * x_0
        mean_coef1 = (0.5 * (F.logsigmoid(logsnr_s) - logsnr_t) + logr).exp()
        mean_coef2 = torch.sigmoid(logsnr_s).sqrt()
    else:
        # E[x_s|x_t] = mean_coef1 * x_t + mean_coef2 * x_0
        mean_coef1 = (logr + log_alpha_st).exp()
        mean_coef2 = (log_one_minus_r + 0.5 * F.logsigmoid(logsnr_s)).exp()

    # strictly speaking, only when var_type == "small",
    # does `logvar` calculated here represent the logarithm
    # of the true posterior variance
    if var_type == "fixed_large":
        logvar = log_one_minus_r + F.logsigmoid(-logsnr_t)
    elif var_type == "fixed_small":
        logvar = log_one_minus_r + F.logsigmoid(-logsnr_s)
    elif var_type == "fixed_medium":
        # linear interpolation in log-space
        assert isinstance(intp_frac, (float, torch.Tensor))
        # logvar = (1 - intp_frac) * logvar_min + intp_frac * logvar_max
        logvar_min = log_one_minus_r + F.logsigmoid(-logsnr_s)
        logvar_max = log_one_minus_r + F.logsigmoid(-logsnr_t)
        logvar = logvar_min.lerp(logvar_max, intp_frac)
    else:
        raise NotImplementedError(var_type)

    return tuple(map(lambda x: x.to(torch.float32), (mean_coef1, mean_coef2, logvar)))


DEBUG = False


def logsnr_to_posterior_ddim(logsnr_s, logsnr_t, eta: float = 0., x0eps_coef: bool = False):
    # upcast to double precision to reduce precision loss
    logsnr_s, logsnr_t = (logsnr_s.to(torch.float64), logsnr_t.to(torch.float64))

    if not DEBUG and eta == 1.:
        return logsnr_to_posterior(logsnr_s, logsnr_t, "fixed_small")
    else:
        if DEBUG:
            print("Debugging mode...")
        logr = logsnr_t - logsnr_s
        if eta == 0:
            log_one_minus_sqrt_r = stable_log1mexp(0.5 * logr)
            if x0eps_coef:
                mean_coef1 = F.logsigmoid(-logsnr_s).mul_(0.5)
                mean_coef2 = F.logsigmoid(logsnr_s).mul_(0.5)
            else:
                mean_coef1 = (F.logsigmoid(-logsnr_s) - F.logsigmoid(-logsnr_t)).mul(0.5).exp()
                mean_coef2 = (log_one_minus_sqrt_r + 0.5 * F.logsigmoid(logsnr_s)).exp()
            logvar = torch.as_tensor(-torch.inf)
        else:
            log_one_minus_r = stable_log1mexp(logr)
            logvar = log_one_minus_r + F.logsigmoid(-logsnr_s) + 2 * math.log(eta)
            if x0eps_coef:
                mean_coef1 = stable_log1mexp(2 * math.log(eta) + log_one_minus_r).add_(
                    F.logsigmoid(-logsnr_s)).mul_(0.5)
                mean_coef2 = F.logsigmoid(logsnr_s).mul_(0.5)
            else:
                mean_coef1 = stable_log1mexp(2 * math.log(eta) + log_one_minus_r).add_(
                    F.logsigmoid(-logsnr_s) - F.logsigmoid(-logsnr_t)).mul_(0.5)
                mean_coef2 = stable_log1mexp((
                    logr + stable_log1mexp(2 * math.log(eta) + log_one_minus_r)
                ).mul_(0.5)).add_(0.5 * F.logsigmoid(logsnr_s))
            mean_coef1, mean_coef2 = mean_coef1.exp_(), mean_coef2.exp_()

        return tuple(map(lambda x: x.to(torch.float32), (mean_coef1, mean_coef2, logvar)))


@torch.jit.script
def pred_x0_from_eps(x_t, eps, logsnr_t):
    return x_t.mul(torch.sigmoid(logsnr_t).rsqrt()) - eps.mul(logsnr_t.neg().mul(.5).exp())


def pred_x0_from_x0eps(x_t, x0eps, logsnr_t):
    x_0, eps = x0eps.chunk(2, dim=1)
    _x_0 = pred_x0_from_eps(x_t, eps, logsnr_t)
    return x_0.mul(torch.sigmoid(-logsnr_t)) + _x_0.mul(torch.sigmoid(logsnr_t))


@torch.jit.script
def pred_eps_from_x0(x_t, x_0, logsnr_t):
    return x_t.mul(torch.sigmoid(-logsnr_t).rsqrt()) - x_0.mul(logsnr_t.mul(.5).exp())


@torch.jit.script
def pred_v_from_x0eps(x_0, eps, logsnr_t):
    return -x_0.mul(torch.sigmoid(-logsnr_t).sqrt()) + eps.mul(torch.sigmoid(logsnr_t).sqrt())


@torch.jit.script
def pred_v_from_x0(x_t, x_0, logsnr_t):
    return x_t.mul(logsnr_t.mul(.5).exp()) - x_0.mul(torch.sigmoid(-logsnr_t).rsqrt())


@torch.jit.script
def pred_x0_from_v(x_t, v, logsnr_t):
    return x_t.mul(torch.sigmoid(logsnr_t).sqrt()) - v.mul(torch.sigmoid(-logsnr_t).sqrt())


@torch.jit.script
def pred_eps_from_v(x_t, v, logsnr_t):
    return x_t.mul(torch.sigmoid(-logsnr_t).sqrt()) + v.mul(torch.sigmoid(logsnr_t).sqrt())


def q_sample(x_0, logsnr_t, eps=None):
    if eps is None:
        eps = torch.randn_like(x_0)
    return x_0.mul(torch.sigmoid(logsnr_t).sqrt()) + eps.mul(torch.sigmoid(-logsnr_t).sqrt())


@torch.jit.script
def q_mean_var(x_0, logsnr_t):
    return x_0.mul(torch.sigmoid(logsnr_t).sqrt()), F.logsigmoid(-logsnr_t)


def raise_error_with_msg(msg):
    def raise_error(*args, **kwargs):
        raise NotImplementedError(msg)

    return raise_error


class GaussianDiffusion:
    def __init__(
            self,
            logsnr_fn,
            sample_timesteps,
            model_out_type,
            model_var_type,
            reweight_type,
            loss_type,
            intp_frac=None,
            w_guide=0.1,
            p_uncond=0.1,
            x0eps_coef=False,
    ):
        self.logsnr_fn = logsnr_fn
        self.sample_timesteps = sample_timesteps

        self.model_out_type = model_out_type
        self.model_var_type = model_var_type
        self.model_var_type = model_var_type

        # from mse_target to re-weighting strategy
        # x0 -> constant
        # eps -> SNR
        # both -> snr_trunc, i.e. max(1, SNR)
        # v -> snr_1plus, i.e. 1+SNR
        self.reweight_type = reweight_type
        self.loss_type = loss_type
        self.intp_frac = intp_frac
        self.w_guide = w_guide
        self.p_uncond = p_uncond
        self.x0eps_coef = x0eps_coef

    def t2logsnr(self, *ts, x=None):
        _broadcast_to = lambda t: broadcast_to(self.logsnr_fn(t), x=x)
        return tuple(map(_broadcast_to, ts))

    def q_posterior_mean_var(
            self, x_0, x_t, logsnr_s, logsnr_t, model_var_type=None, intp_frac=None):
        # x_t here is either referring to the noisy data or the predicted noise
        if model_var_type is None:
            model_var_type = self.model_var_type
        if intp_frac is None:
            intp_frac = self.intp_frac
        mean_coef1, mean_coef2, posterior_logvar = logsnr_to_posterior(
            logsnr_s, logsnr_t,
            var_type=model_var_type, intp_frac=intp_frac, x0eps_coef=self.x0eps_coef)
        posterior_mean = mean_coef1 * x_t + mean_coef2 * x_0
        return posterior_mean, posterior_logvar

    def q_posterior_mean_var_ddim(self, x_0, x_t, logsnr_s, logsnr_t):
        # x_t here is either referring to the noisy data or the predicted noise
        mean_coef1, mean_coef2, posterior_logvar = logsnr_to_posterior_ddim(
            logsnr_s, logsnr_t, eta=0., x0eps_coef=self.x0eps_coef)
        posterior_mean = mean_coef1 * x_t + mean_coef2 * x_0
        return posterior_mean, posterior_logvar

    def p_mean_var(
            self, model_out, x_t, logsnr_s, logsnr_t, clip_denoised, return_pred, use_ddim=False):

        if self.model_var_type == "learned":
            out, intp_frac = model_out.chunk(2, dim=1)
            intp_frac = torch.sigmoid(intp_frac)  # re-scale to (0, 1)
        else:
            intp_frac = None

        # calculate the mean estimate
        _clip = (lambda x: x.clamp(-1., 1.)) if clip_denoised else (lambda x: x)
        _raise_error = raise_error_with_msg(self.model_out_type)
        pred_x_0 = _clip({
                             "x0": lambda arg1, arg2, arg3: arg2,
                             "eps": pred_x0_from_eps,
                             "both": pred_x0_from_x0eps,
                             "v": pred_x0_from_v
                         }.get(self.model_out_type, _raise_error)(x_t, model_out, logsnr_t))
        if self.x0eps_coef:
            if clip_denoised or self.model_out_type != "eps":
                eps = pred_eps_from_x0(x_t, pred_x_0, logsnr_t)
            else:
                eps = model_out
            x_t = eps
        if use_ddim:
            model_mean, model_logvar = self.q_posterior_mean_var_ddim(
                x_0=pred_x_0, x_t=x_t,
                logsnr_s=logsnr_s, logsnr_t=logsnr_t)
        else:
            model_mean, model_logvar = self.q_posterior_mean_var(
                x_0=pred_x_0, x_t=x_t,
                logsnr_s=logsnr_s, logsnr_t=logsnr_t, intp_frac=intp_frac)

        if return_pred:
            return model_mean, model_logvar, pred_x_0
        else:
            return model_mean, model_logvar

    # === sample ===

    def p_sample_step(
            self, denoise_fn, x_t, step, y, generator=None,
            clip_denoised=True, return_pred=False, use_ddim=False):
        s, t = step.div(self.sample_timesteps), \
            step.add(1).div(self.sample_timesteps)
        logsnr_s, logsnr_t = self.t2logsnr(s, t, x=x_t)
        cond = broadcast_to(step > 0, x_t, dtype=torch.bool)
        use_cfg = self.w_guide and y is not None
        if use_cfg:
            x_t = torch.cat([x_t, x_t], dim=0)
            t = torch.cat([t, t], dim=0)
            y = torch.cat([y, torch.zeros_like(y)])
        model_out = denoise_fn(x_t, t, y)
        model_mean, model_logvar, pred_x_0 = self.p_mean_var(
            model_out, x_t, logsnr_s, logsnr_t,
            clip_denoised=clip_denoised, return_pred=True, use_ddim=use_ddim)
        model_mean = torch.where(cond, model_mean, pred_x_0)
        if use_cfg:
            # classifier-free guidance
            model_out, _model_out = model_out.chunk(2, dim=0)
            pred_x_0, _pred_x_0 = pred_x_0.chunk(2, dim=0)
            _model_mean, _, _pred_x_0 = self.p_mean_var(
                _model_out, x_t, logsnr_s, logsnr_t,
                clip_denoised=clip_denoised, return_pred=True, use_ddim=use_ddim)
            _model_mean = torch.where(cond, _model_mean, _pred_x_0)
            model_mean += self.w_guide * (model_mean - _model_mean)
            pred_x_0 += self.w_guide * (pred_x_0 - _pred_x_0)

        noise = torch.empty_like(x_t).normal_(generator=generator)
        sample = model_mean + cond.float() * torch.exp(0.5 * model_logvar) * noise

        return (sample, pred_x_0) if return_pred else sample

    @torch.inference_mode()
    def p_sample(
            self, denoise_fn, shape, noise=None, label=None,
            device="cpu", seed=None, use_ddim=False):
        B = shape[0]
        t = torch.empty((B,), device=device, dtype=torch.float64)
        if seed is None:
            generator = None
        else:
            generator = torch.Generator(device).manual_seed(seed)
        if noise is None:
            x_t = torch.randn(shape, device=device, generator=generator)
        else:
            x_t = noise.to(device)
        if label is not None:
            label = label.to(device)
        for ti in reversed(range(self.sample_timesteps)):
            t.fill_(ti)
            x_t = self.p_sample_step(
                denoise_fn, x_t, step=t, y=label, generator=generator, use_ddim=use_ddim)
        return x_t.cpu()

    @torch.inference_mode()
    def p_sample_progressive(
            self, denoise_fn, shape, noise=None, label=None,
            device="cpu", seed=None, use_ddim=False, pred_freq=50):
        B = shape[0]
        t = torch.empty(B, device=device)
        if seed is None:
            generator = None
        else:
            generator = torch.Generator(device).manual_seed(seed)
        if noise is None:
            x_t = torch.randn(shape, device=device, generator=generator)
        else:
            x_t = noise.to(device)
        L = self.sample_timesteps // pred_freq
        preds = torch.zeros((L, B) + shape[1:], dtype=torch.float32)
        idx = L
        for ti in reversed(range(self.sample_timesteps)):
            t.fill_(ti)
            x_t, pred = self.p_sample_step(
                denoise_fn, x_t, step=t, y=label, generator=generator,
                return_pred=True, use_ddim=use_ddim)
            if (ti + 1) % pred_freq == 0:
                idx -= 1
                preds[idx] = pred.cpu()
        return x_t.cpu(), preds

    # === log likelihood ===
    # bpd: bits per dimension

    def _loss_term_bpd(
            self, model_out, x_0, x_t, logsnr_s, logsnr_t, clip_denoised, return_pred=False):
        # calculate L_t
        # t = 0: negative log likelihood of decoder, -log p(x_0 | x_1)
        # t > 0: variational lower bound loss term, KL term
        true_mean, true_logvar = self.q_posterior_mean_var(
            x_0=x_0, x_t=x_t,
            logsnr_s=logsnr_s, logsnr_t=logsnr_t, model_var_type="fixed_small")
        model_mean, model_logvar, pred_x_0 = self.p_mean_var(
            model_out, x_t=x_t, logsnr_s=logsnr_s, logsnr_t=logsnr_t,
            clip_denoised=clip_denoised, return_pred=True, use_ddim=False)
        kl = normal_kl(true_mean, true_logvar, model_mean, model_logvar)
        kl = flat_mean(kl) / math.log(2.)  # natural base to base 2
        decoder_nll = discretized_gaussian_loglik(
            x_0, pred_x_0, log_scale=0.5 * model_logvar).neg()
        decoder_nll = flat_mean(decoder_nll) / math.log(2.)
        # output = torch.where(s.to(kl.device) > 0, kl, decoder_nll)
        output = (kl, decoder_nll)
        return (*output, pred_x_0) if return_pred else output

    def from_model_out_to_pred(self, x_t, model_out, logsnr_t):
        assert self.model_out_type in {"x0", "eps", "both", "v"}
        if self.model_out_type == "v":
            v = model_out
            x_0 = pred_x0_from_v(x_t, v, logsnr_t)
            eps = pred_eps_from_v(x_t, v, logsnr_t)
            # The above is more accurate and direct. Why do we need the following?
            # uncomment to replicate `google-research/ddpm_w_distillation`
            # eps = pred_eps_from_x0(x_t, x_0, logsnr_t)
        else:
            if self.model_out_type == "x0":
                x_0 = model_out
                eps = pred_eps_from_x0(x_t, x_0, logsnr_t)
            elif self.model_out_type == "eps":
                eps = model_out
                x_0 = pred_x0_from_eps(x_t, eps, logsnr_t)
            elif self.model_out_type == "both":
                # x_0, eps = model_out.chunk(2, dim=1)
                # Why do we need the following?
                x_0 = pred_x0_from_x0eps(x_t, model_out, logsnr_t)
                eps = pred_eps_from_x0(x_t, x_0, logsnr_t)
            else:
                raise NotImplementedError(self.model_out_type)
            v = pred_v_from_x0eps(x_0, eps, logsnr_t)
        return {"constant": x_0, "snr": eps, "snr_trunc": (x_0, eps), "snr_1plus": v}

    def train_loss(self, denoise_fn, x_0, t, y, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        s = None
        if self.loss_type == "kl":
            t = torch.ceil(t * self.sample_timesteps).div(self.sample_timesteps)
            s = t.sub(1 / self.sample_timesteps).clamp(min=0.)
            use_kl = s != 0

        # calculate the loss
        # kl: un-weighted
        # mse: re-weighted

        logsnr_t = self.t2logsnr(t, x=x_0)[0]
        x_t = q_sample(x_0, logsnr_t, eps=noise)
        model_out = denoise_fn(x_t, t, y)

        if self.loss_type == "kl":
            logsnr_s = self.t2logsnr(s, x=x_0)[0]
            kl, nll = self._loss_term_bpd(
                    model_out, x_0=x_0, x_t=x_t, logsnr_s=logsnr_s, logsnr_t=logsnr_t,
                    clip_denoised=False, return_pred=False)
            loss = torch.where(use_kl, kl, nll)  # noqa

        elif self.loss_type == "mse":
            assert self.model_var_type != "learned"
            assert self.reweight_type in {"constant", "snr", "snr_trunc", "snr_1plus"}
            target = {
                "constant": x_0,
                "snr": noise,
                "snr_trunc": (x_0, noise),
                "snr_1plus": pred_v_from_x0eps(x_0, noise, logsnr_t)
            }[self.reweight_type]

            if self.p_uncond and y is not None:
                y *= broadcast_to(
                    torch.rand((y.shape[0],)) > self.p_uncond, y)

            predict = self.from_model_out_to_pred(
                x_t, model_out, logsnr_t
            )[self.reweight_type]

            if isinstance(target, tuple):
                assert len(target) == 2
                loss = torch.maximum(*[
                    flat_mean((tgt - pred).pow(2))
                    for tgt, pred in zip(target, predict)])
            else:
                loss = flat_mean((target - model_out).pow(2))
        else:
            raise NotImplementedError(self.loss_type)

        return loss

    def _prior_bpd(self, x_0):
        B = x_0.shape[0]
        t = torch.ones([B, ], dtype=torch.float32)
        logsnr_t, = self.t2logsnr(t, x=x_0)[0]
        T_mean, T_logvar = q_mean_var(x_0=x_0, logsnr_t=logsnr_t)
        kl_prior = normal_kl(T_mean, T_logvar, mean2=0., logvar2=0.)
        return flat_mean(kl_prior) / math.log(2.)

    def calc_all_bpd(self, denoise_fn, x_0, y, clip_denoised=True):
        B, T = x_0.shape, self.sample_timesteps
        s = torch.empty([B, ], dtype=torch.float64)
        t = torch.empty([B, ], dtype=torch.float64)
        loss = torch.zeros([B, T], dtype=torch.float32)
        mse = torch.zeros([B, T], dtype=torch.float32)

        for i in range(T - 1, -1, -1):
            s.fill_(i / self.sample_timesteps)
            t.fill_((i + 1) / self.sample_timesteps)
            logsnr_s, logsnr_t = self.t2logsnr(s, t, x=x_0)
            x_t = q_sample(x_0, logsnr_t=logsnr_t)
            model_out = denoise_fn(x_t, t, y)
            loss_term, pred_x_0 = self._loss_term_bpd(
                model_out, x_0, x_t=x_t, logsnr_s=logsnr_s, logsnr_t=logsnr_t,
                clip_denoised=clip_denoised, return_pred=True)
            loss[:, i] = loss_term
            mse[:, i] = flat_mean((pred_x_0 - x_0).pow(2))

        prior_bpd = self._prior_bpd(x_0)
        total_bpd = torch.sum(loss, dim=1) + prior_bpd
        return total_bpd, loss, prior_bpd, mse


if __name__ == "__main__":
    DEBUG = True


    def test_logsnr_to_posterior():
        logsnr_schedule = get_logsnr_schedule("cosine")
        logsnr = logsnr_schedule(torch.linspace(0, 1, 1001))
        logsnr_s, logsnr_t = logsnr[:-1], logsnr[1:]
        mean_coef1, mean_coef2, _ = logsnr_to_posterior(logsnr_s, logsnr_t, "fixed_small")
        mean_coef1_, mean_coef2_, _ = logsnr_to_posterior(logsnr_s, logsnr_t, "fixed_small", x0eps_coef=True)
        logr = logsnr_t - logsnr_s
        print(torch.allclose(mean_coef1 * torch.sigmoid(-logsnr_t).sqrt(), mean_coef1_))
        print(torch.allclose(mean_coef2 + torch.sigmoid(logsnr_s).sqrt() * logr.exp(), mean_coef2_))


    def test_logsnr_to_posterior_ddim():
        logsnr_schedule = get_logsnr_schedule("cosine")
        logsnr = logsnr_schedule(torch.linspace(0, 1, 1001))
        logsnr_s, logsnr_t = logsnr[:-1], logsnr[1:]
        if len(logsnr) <= 50:
            print(logsnr)
        mean_coef1, mean_coef2, logvar = logsnr_to_posterior(
            logsnr_s, logsnr_t, "fixed_small")
        mean_coef1_, mean_coef2_, logvar_ = logsnr_to_posterior_ddim(
            logsnr_s, logsnr_t, eta=1.)
        print(
            torch.allclose(mean_coef1, mean_coef1_),
            torch.allclose(mean_coef2, mean_coef2_),
            torch.allclose(logvar, logvar_),
        )
        mean_coef1, mean_coef2, logvar = logsnr_to_posterior_ddim(
            logsnr_s, logsnr_t, eta=0.5)
        mean_coef1_, mean_coef2_, logvar_ = logsnr_to_posterior_ddim(
            logsnr_s, logsnr_t, eta=0.5, x0eps_coef=True)
        print(torch.allclose(mean_coef1 * torch.sigmoid(-logsnr_t).sqrt(), mean_coef1_))
        print(torch.allclose(mean_coef2 + torch.sigmoid(logsnr_s).sqrt() * mean_coef1, mean_coef2_))

    def test_legacy():
        logsnr_schedule = get_logsnr_schedule("legacy")
        t = torch.linspace(0, 1, 1000, dtype=torch.float32)
        alphas = torch.sigmoid(logsnr_schedule(t))
        betas = torch.linspace(0.0001, 0.02, 1000)
        _alphas = torch.cumprod(1 - betas, dim=0)
        print((alphas - _alphas).abs().max())
        print((alphas - _alphas).div(_alphas).abs().max())


    def test_schedule():
        logsnr_schedule = get_logsnr_schedule("cosine", rescale=True)
        length = 50
        t = torch.linspace(0, 1, 1001, dtype=torch.float32)[np.linspace(0, 1000, length)]
        logsnr = logsnr_schedule(t)

        def cosine(logsnr, return_grad=False):
            if return_grad and not logsnr.requires_grad:
                logsnr.requires_grad_(True)
            method1 = torch.sigmoid(logsnr).sqrt()
            method2 = logsnr.exp().sqrt() * torch.sigmoid(-logsnr).sqrt()
            if return_grad:
                method1 = torch.autograd.grad(method1.sum(), inputs=[logsnr, ])[0]
                method2 = torch.autograd.grad(method2.sum(), inputs=[logsnr, ])[0]
            return method1, method2

        def sine(logsnr, return_grad=False):
            if return_grad and not logsnr.requires_grad:
                logsnr.requires_grad_(True)
            method1 = torch.sigmoid(-logsnr).sqrt()
            method2 = logsnr.neg().exp().sqrt() * torch.sigmoid(logsnr).sqrt()
            if return_grad:
                method1 = torch.autograd.grad(method1.sum(), inputs=[logsnr, ])[0]
                method2 = torch.autograd.grad(method2.sum(), inputs=[logsnr, ])[0]
            return method1, method2

        def cotangent(logsnr, return_grad=False):
            if return_grad and not logsnr.requires_grad:
                logsnr.requires_grad_(True)
            method1 = logsnr.mul(0.5).exp()
            method2 = torch.sigmoid(logsnr).sqrt() * torch.sigmoid(-logsnr).rsqrt()
            if return_grad:
                method1 = torch.autograd.grad(method1.sum(), inputs=[logsnr, ])[0]
                method2 = torch.autograd.grad(method2.sum(), inputs=[logsnr, ])[0]
            return method1, method2

        def check_allclose(fn, args):
            val1, val2 = fn(*args)
            all_close = torch.allclose(val1, val2)
            print(fn.__name__, end=" ")
            if all_close:
                print("Congrats, all close!")
            else:
                print("Uh oh, no luck...")
                if len(val1) <= 50:
                    print(val1, val2)
                print("abs. err.:", (val1 - val2).abs().max())
                print("rel. err.:", (val1 - val2).div(val1 + 1e-12).abs().max())

        at = torch.atan(logsnr.clamp(-20, 20).mul(-0.5).exp()).div(0.5 * torch.pi)
        if length <= 50:
            print(logsnr)
            print(torch.allclose(t, at))
        # check numerical precision
        check_allclose(cosine, (logsnr,))
        check_allclose(sine, (logsnr,))
        check_allclose(cotangent, (logsnr,))
        # check gradient precision
        check_allclose(cosine, (logsnr, True))
        check_allclose(sine, (logsnr, True))
        check_allclose(cotangent, (logsnr, True))


    # run tests
    TESTS = [
        test_logsnr_to_posterior,  # 0
        test_logsnr_to_posterior_ddim,  # 1
        test_legacy,  # 2
        test_schedule,  # 3
    ]
    TEST_INDICES = [0, 1]
    for i in TEST_INDICES:
        TESTS[i]()
