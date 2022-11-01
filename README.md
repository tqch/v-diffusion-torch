# PyTorch Implementation of V-objective Diffusion Probabilistic Model (VDPM) and more

## Key features
- [x] improved UNet design (conditioning, resampling, etc.) [^1]
- [x] continuous-time training on log-SNR schedule [^2]
- [x] DDIM sampler [^3]
- [x] MSE loss reweighting (constant, SNR, truncated-SNR) [^4]
- [x] velocity prediction [^4]
- [x] classifier-free guidance [^5]
- [x] others:
	- [x] distributed data parallel (multi-gpu training)
	- [x] gradient accumulation
	- [x] FID/Precision/Recall evaluation

## Conditional generation

### CIFAR-10

<p align="center">
	<table>
        <tr>
            <th>guidance strength</th>
            <th>class</th>
            <th>images</th>
        </tr><tr>
            <td rowspan="10"><b>w=0</b></td>
            <td>airplanes</td>
            <td rowspan="10"><img alt="w=0" src="./assets/cifar10_w0.jpg" height="100%" width="100%" />
        </tr><tr>
            <td>cars</td>
        </tr><tr>
            <td>birds</td>
        </tr><tr>
            <td>cats</td>
        </tr><tr>
            <td>deer</td>
        </tr><tr>
            <td>dogs</td>
        </tr><tr>
            <td>frogs</td>
        </tr><tr>
            <td>horses</td>
        </tr><tr>
            <td>ships</td>
        </tr><tr>
            <td>trucks</td>
        </tr><tr>
            <td rowspan="10"><b>w=1</b></td>
            <td>airplanes</td>
            <td rowspan="10"><img alt="w=0" src="./assets/cifar10_w1.jpg" height="100%" width="100%" />
        </tr><tr>
            <td>cars</td>
        </tr><tr>
            <td>birds</td>
        </tr><tr>
            <td>cats</td>
        </tr><tr>
            <td>deer</td>
        </tr><tr>
            <td>dogs</td>
        </tr><tr>
            <td>frogs</td>
        </tr><tr>
            <td>horses</td>
        </tr><tr>
            <td>ships</td>
        </tr><tr>
            <td>trucks</td>
        </tr><tr>
            <td rowspan="10"><b>w=3</b></td>
			<td>airplanes</td>
			<td rowspan="10"><img alt="w=0" src="./assets/cifar10_w3.jpg" height="100%" width="100%" />
        </tr><tr>
            <td>cars</td>
        </tr><tr>
            <td>birds</td>
        </tr><tr>
            <td>cats</td>
        </tr><tr>
            <td>deer</td>
        </tr><tr>
            <td>dogs</td>
        </tr><tr>
            <td>frogs</td>
        </tr><tr>
            <td>horses</td>
        </tr><tr>
            <td>ships</td>
        </tr><tr>
            <td>trucks</td>
        </tr>
	</table>
</p>

### CelebA

<p align="center">
	<table>
        <tr>
            <th>guidance strength</th>
            <th>tag</th>
            <th>Black_Hair</th>
            <th>Blond_Hair</th>
            <th>Brown_Hair</th>
            <th>Gray_Hair</th>
        </tr><tr>
        	<td rowspan="5"><b>w=0</b></td>
            <td>Receding_Hairline</td>
            <td rowspan="5" colspan="4"><img alt="w=0" src="./assets/celeba_w0.jpg" height="100%" width="100%" />
        </tr><tr>
			<td>Straight_Hair</td>
        </tr><tr>
			<td>Wavy_Hair</td>
        </tr><tr>
            <td>Bald</td>
        </tr><tr>
            <td>Bangs</td>
        </tr><tr>
        	<td rowspan="5"><b>w=1</b></td>
            <td>Receding_Hairline</td>
            <td rowspan="5" colspan="4"><img alt="w=0" src="./assets/celeba_w1.jpg" height="100%" width="100%" />
        </tr><tr>
			<td>Straight_Hair</td>
        </tr><tr>
			<td>Wavy_Hair</td>
        </tr><tr>
            <td>Bald</td>
        </tr><tr>
            <td>Bangs</td>
        </tr><tr>
        	<td rowspan="5"><b>w=3</b></td>
            <td>Receding_Hairline</td>
            <td rowspan="5" colspan="4"><img alt="w=3" src="./assets/celeba_w3.jpg" height="100%" width="100%" />
        </tr><tr>
			<td>Straight_Hair</td>
        </tr><tr>
			<td>Wavy_Hair</td>
        </tr><tr>
            <td>Bald</td>
        </tr><tr>
            <td>Bangs</td>
        </tr>
	</table>
</p>

## Acknowledgement
The development of this codebase is largely based on the official JAX implementation open-sourced by *Google Research* and my previous PyTorch implementation of DDPM, which are available at [[google-research/diffusion_distillation]](https://github.com/google-research/google-research/tree/master/diffusion_distillation/diffusion_distillation) and [[tqch/ddpm-torch]](https://github.com/tqch/ddpm-torch) respectively.

## References

[^1]: Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." Advances in Neural Information Processing Systems 33 (2020): 6840-6851.
[^2]: Kingma, Diederik, Tim Salimans, Ben Poole, and Jonathan Ho. "Variational diffusion models." Advances in neural information processing systems 34 (2021): 21696-21707.
[^3]: Song, Jiaming, Chenlin Meng, and Stefano Ermon. "Denoising diffusion implicit models." arXiv preprint arXiv:2010.02502 (2020).
[^4]: Salimans, Tim, and Jonathan Ho. "Progressive distillation for fast sampling of diffusion models." arXiv preprint arXiv:2202.00512 (2022).
[^5]: Ho, Jonathan, Tim Salimans. ‘Classifier-Free Diffusion Guidance’. NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. https://openreview.net/forum?id=qw8AKxfYbI.

