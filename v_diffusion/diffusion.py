import math
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


def get_logsnr_schedule(schedule, logsnr_min: float = -20., logsnr_max: float = 20.):
    """
    schedule is named according to the relationship between alpha2 and t,
    i.e. alpha2 as a XX function of affine transformation of t (except for legacy)
    """

    logsnr_min, logsnr_max = torch.as_tensor(logsnr_min), torch.as_tensor(logsnr_max)
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
    elif schedule == "legacy":
        """
        continuous version of the (discrete) linear schedule used by \
          Ho, Jonathan, Ajay Jain, and Pieter Abbeel. \
            "Denoising diffusion probabilistic models." \
              Advances in Neural Information Processing Systems 33 (2020): 6840-6851.
        """
        delta_max, delta_min = (
            torch.as_tensor(1 - 0.0001),
            torch.as_tensor(1 - 0.02))
        delta_max_m1 = torch.as_tensor(-0.0001)
        log_delta_max = torch.log1p(delta_max_m1)
        log_delta_min = torch.log1p(torch.as_tensor(-0.02))
        delta_range = delta_max - delta_min
        log_alpha_range = (delta_max * log_delta_max -
                           delta_min * log_delta_min) / delta_range - 1

        def schedule_fn(t):
            tau = delta_max - delta_range * t
            tau_m1 = delta_max_m1 - delta_range * t
            log_alpha = (
                    (delta_max * log_delta_max - tau * torch.log1p(tau_m1))
                    / delta_range - t).mul(-20. / log_alpha_range).add(-2.0612e-09)
            return log_alpha - stable_log1mexp(log_alpha)

        return schedule_fn

    else:
        raise NotImplementedError
    b = logsnr2t(logsnr_max)
    a = logsnr2t(logsnr_min) - b

    def schedule_fn(t):
        _a, _b = broadcast_to(a, t), broadcast_to(b, t)
        return t2logsnr(_a * t + _b)

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


def logsnr_to_posterior(logsnr_s, logsnr_t, var_type: str, intp_frac=None):
    # upcast to double precision to reduce precision loss
    logsnr_s, logsnr_t = (
        logsnr_s.to(torch.float64), logsnr_t.to(torch.float64))

    log_alpha_st = 0.5 * (F.logsigmoid(logsnr_s) - F.logsigmoid(logsnr_t))
    logr = logsnr_t - logsnr_s
    log_one_minus_r = stable_log1mexp(logr)
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
        logvar = (
                intp_frac * (log_one_minus_r + F.logsigmoid(-logsnr_t)) +
                (1. - intp_frac) * (log_one_minus_r + F.logsigmoid(-logsnr_s))
        )
    else:
        raise NotImplementedError(var_type)

    return tuple(map(lambda x: x.to(torch.float32), (mean_coef1, mean_coef2, logvar)))


DEBUG = False


def logsnr_to_posterior_ddim(logsnr_s, logsnr_t, eta: float):
    # upcast to double precision to reduce precision loss
    logsnr_s, logsnr_t = (
        logsnr_s.to(torch.float64), logsnr_t.to(torch.float64))

    if not DEBUG and eta == 1.:
        return logsnr_to_posterior(logsnr_s, logsnr_t, "fixed_small")
    else:
        if DEBUG:
            print("Debugging mode...")
        log_alpha_st = 0.5 * (F.logsigmoid(logsnr_s) - F.logsigmoid(logsnr_t))
        logr = logsnr_t - logsnr_s
        if eta == 0:
            log_one_minus_sqrt_r = stable_log1mexp(0.5 * logr)
            mean_coef1 = (F.logsigmoid(-logsnr_s) - F.logsigmoid(-logsnr_t)).mul(0.5).exp()
            mean_coef2 = (log_one_minus_sqrt_r + 0.5 * F.logsigmoid(logsnr_s)).exp()
            logvar = torch.as_tensor(-torch.inf)
        else:
            log_one_minus_r = stable_log1mexp(logr)
            logvar = log_one_minus_r + F.logsigmoid(-logsnr_s) + 2 * math.log(eta)
            mean_coef1 = stable_log1mexp(
                logvar - F.logsigmoid(-logsnr_s))
            mean_coef1 += F.logsigmoid(-logsnr_s) - F.logsigmoid(-logsnr_t)
            mean_coef1 *= 0.5
            mean_coef2 = stable_log1mexp(mean_coef1 - log_alpha_st).add(
                0.5 * F.logsigmoid(logsnr_s))
            mean_coef1, mean_coef2 = mean_coef1.exp(), mean_coef2.exp()

        return tuple(map(lambda x: x.to(torch.float32), (mean_coef1, mean_coef2, logvar)))


@torch.jit.script
def pred_x0_from_eps(x_t, eps, logsnr_t):
    return x_t.div(torch.sigmoid(logsnr_t).sqrt()) - eps.mul(logsnr_t.neg().mul(.5).exp())


def pred_x0_from_x0eps(x_t, x0eps, logsnr_t):
    x_0, eps = x0eps.chunk(2, dim=1)
    _x_0 = pred_x0_from_eps(x_t, eps, logsnr_t)
    return x_0.mul(torch.sigmoid(-logsnr_t)) + _x_0.mul(torch.sigmoid(logsnr_t))


@torch.jit.script
def pred_eps_from_x0(x_t, x_0, logsnr_t):
    return x_t.mul(torch.sigmoid(-logsnr_t).sqrt()) - x_0.mul(logsnr_t.mul(.5).exp())


@torch.jit.script
def pred_v_from_x0eps(x_0, eps, logsnr_t):
    return -x_0.mul(torch.sigmoid(-logsnr_t).sqrt()) + eps.mul(torch.sigmoid(logsnr_t).sqrt())


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
            p_uncond=0.1
    ):
        self.logsnr_fn = logsnr_fn
        self.sample_timesteps = sample_timesteps

        self.model_out_type = model_out_type
        self.model_var_type = model_var_type

        # from mse_target to re-weighting strategy
        # x0 -> constant
        # eps -> SNR
        # both -> truncated_SNR, i.e. max(1, SNR)
        self.reweight_type = reweight_type
        self.loss_type = loss_type
        self.intp_frac = intp_frac
        self.w_guide = w_guide
        self.p_uncond = p_uncond

    def t2logsnr(self, *ts, x=None):
        _broadcast_to = lambda t: broadcast_to(
            self.logsnr_fn(t), x=x)
        return tuple(map(_broadcast_to, ts))

    def q_posterior_mean_var(
            self, x_0, x_t, logsnr_s, logsnr_t, model_var_type=None, intp_frac=None):
        model_var_type = model_var_type or self.model_var_type
        intp_frac = self.intp_frac or intp_frac
        mean_coef1, mean_coef2, posterior_logvar = logsnr_to_posterior(
            logsnr_s, logsnr_t, var_type=model_var_type, intp_frac=intp_frac)
        posterior_mean = mean_coef1 * x_t + mean_coef2 * x_0
        return posterior_mean, posterior_logvar

    def q_posterior_mean_var_ddim(self, x_0, x_t, logsnr_s, logsnr_t):
        mean_coef1, mean_coef2, posterior_logvar = logsnr_to_posterior_ddim(
            logsnr_s, logsnr_t, eta=0.)
        posterior_mean = mean_coef1 * x_t + mean_coef2 * x_0
        return posterior_mean, posterior_logvar

    def p_mean_var(
            self, denoise_fn, x_t, s, t, y, clip_denoised, return_pred, use_ddim=False):

        out = denoise_fn(x_t, t, y=y)
        logsnr_s, logsnr_t = self.t2logsnr(s, t, x=x_t)

        if self.model_var_type == "learned":
            out, intp_frac = out.chunk(2, dim=1)
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
                         }.get(self.model_out_type, _raise_error)(x_t, out, logsnr_t))
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
        cond = broadcast_to(step > 0, x_t, dtype=torch.bool)
        model_mean, model_logvar, pred_x_0 = self.p_mean_var(
            denoise_fn, x_t, s, t, y,
            clip_denoised=clip_denoised, return_pred=True, use_ddim=use_ddim)
        model_mean = torch.where(cond, model_mean, pred_x_0)
        if self.w_guide and y is not None:
            # classifier-free guidance
            _model_mean, _, _pred_x_0 = self.p_mean_var(
                denoise_fn, x_t, s, t, torch.zeros_like(y),
                clip_denoised=clip_denoised, return_pred=True, use_ddim=use_ddim)
            _model_mean = torch.where(cond, _model_mean, _pred_x_0)
            model_mean += self.w_guide * (model_mean - _model_mean)

        noise = torch.empty_like(x_t).normal_(generator=generator)
        sample = model_mean + cond.float() * torch.exp(0.5 * model_logvar) * noise

        return (sample, pred_x_0) if return_pred else sample

    @torch.inference_mode()
    def p_sample(
            self, denoise_fn, shape, noise=None, label=None,
            device="cpu", seed=None, use_ddim=False):
        B = shape[0]
        t = torch.empty((B,), device=device)
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
            self, denoise_fn, x_0, x_t, s, t, y, clip_denoised, return_pred):
        logsnr_s, logsnr_t = self.t2logsnr(s, t, x=x_0)
        # calculate L_t
        # t = 0: negative log likelihood of decoder, -\log p(x_0 | x_1)
        # t > 0: variational lower bound loss term, KL term
        true_mean, true_logvar = self.q_posterior_mean_var(
            x_0=x_0, x_t=x_t,
            logsnr_s=logsnr_s, logsnr_t=logsnr_t, model_var_type="fixed_small")
        model_mean, model_logvar, pred_x_0 = self.p_mean_var(
            denoise_fn, x_t=x_t, s=s, t=t, y=y,
            clip_denoised=clip_denoised, return_pred=True, use_ddim=False)
        kl = normal_kl(true_mean, true_logvar, model_mean, model_logvar)
        kl = flat_mean(kl) / math.log(2.)  # natural base to base 2
        decoder_nll = discretized_gaussian_loglik(
            x_0, pred_x_0, log_scale=0.5 * model_logvar).neg()
        decoder_nll = flat_mean(decoder_nll) / math.log(2.)
        output = torch.where(s.to(kl.device) > 0, kl, decoder_nll)
        return (output, pred_x_0) if return_pred else output

    def from_model_out_to_pred(self, x_t, model_out, logsnr_t):
        assert self.model_out_type in {"x0", "eps", "both", "v"}
        if self.model_out_type == "v":
            v = model_out
            x_0 = pred_x0_from_v(x_t, v, logsnr_t)
            eps = pred_eps_from_v(x_t, v, logsnr_t)
        else:
            if self.model_out_type == "x0":
                x_0 = model_out
                eps = pred_eps_from_x0(x_t, x_0, logsnr_t)
            elif self.model_out_type == "eps":
                eps = model_out
                x_0 = pred_x0_from_eps(x_t, eps, logsnr_t)
            elif self.model_out_type == "both":
                x_0, eps = model_out.chunk(2, dim=1)
            else:
                raise NotImplementedError(self.model_out_type)
            v = pred_v_from_x0eps(x_0, eps, logsnr_t)
        return {"constant": x_0, "snr": eps, "truncated_snr": (x_0, eps), "alpha2": v}

    def train_losses(self, denoise_fn, x_0, t, y, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        s = None
        if self.loss_type == "kl":
            t = torch.ceil(t * self.sample_timesteps)
            s = t.sub(1).div(self.sample_timesteps)
            t = t.div(self.sample_timesteps)

        # calculate the loss
        # kl: un-weighted
        # mse: re-weighted

        logsnr_t, = self.t2logsnr(t, x=x_0)
        x_t = q_sample(x_0, logsnr_t, eps=noise)
        if self.loss_type == "kl":
            losses = self._loss_term_bpd(
                denoise_fn, x_0=x_0, x_t=x_t, s=s, t=t, y=y,
                clip_denoised=False, return_pred=False)
        elif self.loss_type == "mse":
            assert self.model_var_type != "learned"
            assert self.reweight_type in {"constant", "snr", "truncated_snr", "alpha2"}
            target = {
                "constant": x_0,
                "snr": noise,
                "truncated_snr": (x_0, noise),
                "alpha2": pred_v_from_x0eps(x_0, noise, logsnr_t)
            }[self.reweight_type]

            if self.p_uncond and y is not None:
                y *= broadcast_to(
                    torch.rand((y.shape[0],)) > self.p_uncond, y)

            model_out = denoise_fn(x_t, t, y=y)
            predict = self.from_model_out_to_pred(
                x_t, model_out, logsnr_t
            )[self.reweight_type]

            if isinstance(target, tuple):
                assert len(target) == 2
                losses = torch.maximum(*[
                    flat_mean((tgt - pred).pow(2))
                    for tgt, pred in zip(target, predict)])
            else:
                losses = flat_mean((target - model_out).pow(2))
        else:
            raise NotImplementedError(self.loss_type)

        return losses

    def _prior_bpd(self, x_0):
        B = x_0.shape[0]
        t = torch.ones([B, ], dtype=torch.float32)
        logsnr_t, = self.t2logsnr(t, x=x_0)
        T_mean, T_logvar = q_mean_var(x_0=x_0, logsnr_t=logsnr_t)
        kl_prior = normal_kl(T_mean, T_logvar, mean2=0., logvar2=0.)
        return flat_mean(kl_prior) / math.log(2.)

    def calc_all_bpd(self, denoise_fn, x_0, y, clip_denoised=True):
        B, T = x_0.shape, self.sample_timesteps
        s = torch.empty([B, ], dtype=torch.float32)
        t = torch.empty([B, ], dtype=torch.float32)
        losses = torch.zeros([B, T], dtype=torch.float32)
        mses = torch.zeros([B, T], dtype=torch.float32)

        for i in range(T - 1, -1, -1):
            s.fill_(i / self.sample_timesteps)
            t.fill_((i + 1) / self.sample_timesteps)
            logsnr_t, = self.t2logsnr(t)
            x_t = q_sample(x_0, logsnr_t=logsnr_t)
            loss, pred_x_0 = self._loss_term_bpd(
                denoise_fn, x_0, x_t=x_t, s=s, t=t, y=y,
                clip_denoised=clip_denoised, return_pred=True)
            losses[:, i] = loss
            mses[:, i] = flat_mean((pred_x_0 - x_0).pow(2))

        prior_bpd = self._prior_bpd(x_0)
        total_bpd = torch.sum(losses, dim=1) + prior_bpd
        return total_bpd, losses, prior_bpd, mses


if __name__ == "__main__":
    DEBUG = True


    def test_logsnr_to_posterior():
        logsnr_schedule = get_logsnr_schedule("cosine")
        logsnr_s = logsnr_schedule(torch.as_tensor(0.))
        logsnr_t = logsnr_schedule(torch.as_tensor(1. / 1000))
        print(logsnr_to_posterior(logsnr_s, logsnr_t, "fixed_small"))
        logsnr_s = logsnr_schedule(torch.as_tensor(999. / 1000))
        logsnr_t = logsnr_schedule(torch.as_tensor(1.))
        print(logsnr_to_posterior(logsnr_s, logsnr_t, "fixed_small"))


    def test_logsnr_to_posterior_ddim():
        logsnr_schedule = get_logsnr_schedule("cosine")
        t = torch.linspace(0, 1, 1001, dtype=torch.float32)
        print(logsnr_schedule(t))
        logsnr_s = logsnr_schedule(t[:-1])
        logsnr_t = logsnr_schedule(t[1:])
        mean_coef1, mean_coef2, logvar = logsnr_to_posterior(
            logsnr_s, logsnr_t, "fixed_small")
        mean_coef1_, mean_coef2_, logvar_ = logsnr_to_posterior_ddim(
            logsnr_s, logsnr_t, eta=1.)
        print(
            torch.allclose(mean_coef1, mean_coef1_),
            torch.allclose(mean_coef2, mean_coef2_),
            torch.allclose(logvar, logvar_))


    def test_legacy():
        logsnr_schedule = get_logsnr_schedule("legacy")
        t = torch.linspace(0, 1, 1000, dtype=torch.float32)
        print(torch.sigmoid(logsnr_schedule(t))[::10])
        print(logsnr_schedule(t)[::10])
        t = torch.rand(10000, dtype=torch.float32)
        print(logsnr_schedule(t))

    # run tests
    TESTS = [test_logsnr_to_posterior, test_logsnr_to_posterior_ddim, test_legacy]
    TEST_INDICES = []
    for i in TEST_INDICES:
        TESTS[i]()
