import os, sys, math, time, random, datetime, functools
import numpy as np
from pathlib import Path
from loguru import logger
from copy import deepcopy
from omegaconf import OmegaConf
from collections import OrderedDict
from einops import rearrange
from contextlib import nullcontext
import itertools
from monai.networks.nets import UNet
from datapipe.datasets import create_dataset
from math import exp
from torch.autograd import Variable

from utils import util_net
from utils import util_common
from utils import util_image

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.utils.data as udata
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.utils as vutils
from torch.nn.parallel import DistributedDataParallel as DDP


class Sobel_grad():
    def __init__(self, device):
        # Sobel-X
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1,1,3,3)
        # Sobel-Y
        sobel_y = torch.tensor([[-1,-2,-1],
                                [ 0, 0, 0],
                                [ 1, 2, 1]], dtype=torch.float32).view(1,1,3,3)

        self.weight_x = sobel_x.to(device)
        self.weight_y = sobel_y.to(device)

    def grad(self, im):
        # im: [B,1,H,W]  (灰度图，范围 [0,1] 或 [-1,1])
        gx = F.conv2d(im, self.weight_x, padding=1)
        gy = F.conv2d(im, self.weight_y, padding=1)
        grad = torch.sqrt(gx**2 + gy**2 + 1e-6)  # 梯度幅值
        return grad


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def patch_ssim_loss(img_a, img_b, img_f, patch_size=7, ssim=SSIM()):
    mean_a = F.avg_pool2d(img_a, patch_size, stride=1, padding=patch_size // 2)
    mean_b = F.avg_pool2d(img_b, patch_size, stride=1, padding=patch_size // 2)

    std_a = torch.sqrt(F.avg_pool2d(img_a ** 2, patch_size, stride=1, padding=patch_size // 2) - mean_a ** 2 + 1e-6)
    std_b = torch.sqrt(F.avg_pool2d(img_b ** 2, patch_size, stride=1, padding=patch_size // 2) - mean_b ** 2 + 1e-6)

    # 按照局部std选择参考图像
    mask = (std_a > std_b).float()
    pref_img = mask * img_a + (1 - mask) * img_b

    ssim_val = ssim(img_f, pref_img)
    Lssim = 1 - ssim_val

    return Lssim

class TrainerBase:
    def __init__(self, configs):
        self.configs = configs

        # setup distributed training: self.num_gpus, self.rank
        self.setup_dist()

        # setup seed
        self.setup_seed()
        self.ssim = 0

    def setup_dist(self):
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn')
            rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(
                    timeout=datetime.timedelta(seconds=3600),
                    backend='nccl',
                    init_method='env://',
                    )

        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0

    def setup_seed(self, seed=None, global_seeding=None):
        if seed is None:
            seed = self.configs.train.get('seed', 12345)
        if global_seeding is None:
            global_seeding = self.configs.train.global_seeding
            assert isinstance(global_seeding, bool)
        if not global_seeding:
            seed += self.rank
            torch.cuda.manual_seed(seed)
        else:
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def init_logger(self):
        if self.configs.resume:
            assert self.configs.resume.endswith(".pth")
            save_dir = Path(self.configs.resume).parents[1]
            project_id = save_dir.name
        else:
            project_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            save_dir = Path(self.configs.save_dir) / project_id
            if not save_dir.exists() and self.rank == 0:
                save_dir.mkdir(parents=True)

        # setting log counter
        if self.rank == 0:
            self.log_step = {phase: 1 for phase in ['train', 'val']}
            self.log_step_img = {phase: 1 for phase in ['train', 'val']}

        # text logging
        logtxet_path = save_dir / 'training.log'
        if self.rank == 0:
            if logtxet_path.exists():
                assert self.configs.resume
            self.logger = logger
            self.logger.remove()
            self.logger.add(logtxet_path, format="{message}", mode='a', level='INFO')
            self.logger.add(sys.stdout, format="{message}")

        # tensorboard logging
        log_dir = save_dir / 'tf_logs'
        self.tf_logging = self.configs.train.tf_logging
        if self.rank == 0 and self.tf_logging:
            if not log_dir.exists():
                log_dir.mkdir()
            self.writer = SummaryWriter(str(log_dir))

        # checkpoint saving
        ckpt_dir = save_dir / 'ckpts'
        self.ckpt_dir = ckpt_dir
        if self.rank == 0 and (not ckpt_dir.exists()):
            ckpt_dir.mkdir()
        if 'ema_rate' in self.configs.train:
            self.ema_rate = self.configs.train.ema_rate
            assert isinstance(self.ema_rate, float), "Ema rate must be a float number"
            ema_ckpt_dir = save_dir / 'ema_ckpts'
            self.ema_ckpt_dir = ema_ckpt_dir
            if self.rank == 0 and (not ema_ckpt_dir.exists()):
                ema_ckpt_dir.mkdir()

        # save images into local disk
        self.local_logging = self.configs.train.local_logging
        if self.rank == 0 and self.local_logging:
            image_dir = save_dir / 'images'
            if not image_dir.exists():
                (image_dir / 'train').mkdir(parents=True)
                (image_dir / 'val').mkdir(parents=True)
            self.image_dir = image_dir

        # logging the configurations
        if self.rank == 0:
            self.logger.info(OmegaConf.to_yaml(self.configs))

    def close_logger(self):
        if self.rank == 0 and self.tf_logging:
            self.writer.close()

    def resume_from_ckpt(self):
        def _load_ema_state(ema_state, ckpt):
            for key in ema_state.keys():
                if key not in ckpt and key.startswith('module'):
                    ema_state[key] = deepcopy(ckpt[7:].detach().data)
                elif key not in ckpt and (not key.startswith('module')):
                    ema_state[key] = deepcopy(ckpt['module.'+key].detach().data)
                else:
                    ema_state[key] = deepcopy(ckpt[key].detach().data)

        if self.configs.resume:
            assert self.configs.resume.endswith(".pth") and os.path.isfile(self.configs.resume)

            if self.rank == 0:
                self.logger.info(f"=> Loaded checkpoint from {self.configs.resume}")
            ckpt = torch.load(self.configs.resume, map_location=f"cuda:{self.rank}")
            util_net.reload_model(self.model, ckpt['state_dict'])
            torch.cuda.empty_cache()

            # learning rate scheduler
            self.iters_start = ckpt['iters_start']
            for ii in range(1, self.iters_start+1):
                self.adjust_lr(ii)

            # logging
            if self.rank == 0:
                self.log_step = ckpt['log_step']
                self.log_step_img = ckpt['log_step_img']

            # EMA model
            if self.rank == 0 and hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / ("ema_"+Path(self.configs.resume).name)
                self.logger.info(f"=> Loaded EMA checkpoint from {str(ema_ckpt_path)}")
                ema_ckpt = torch.load(ema_ckpt_path, map_location=f"cuda:{self.rank}")
                _load_ema_state(self.ema_state, ema_ckpt)
            torch.cuda.empty_cache()

            # AMP scaler
            if self.amp_scaler is not None:
                if "amp_scaler" in ckpt:
                    self.amp_scaler.load_state_dict(ckpt["amp_scaler"])
                    if self.rank == 0:
                        self.logger.info("Loading scaler from resumed state...")

            # reset the seed
            self.setup_seed(seed=self.iters_start)
        else:
            self.iters_start = 0

    def setup_optimizaton(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.configs.train.lr,
                                           weight_decay=self.configs.train.weight_decay)

        # amp settings
        self.amp_scaler = amp.GradScaler() if self.configs.train.use_amp else None

    def build_model(self):
        params = self.configs.model.get('params', dict)
        model = util_common.get_obj_from_str(self.configs.model.target)(**params)
        model.cuda()
        if self.configs.model.ckpt_path is not None:
            ckpt_path = self.configs.model.ckpt_path
            if self.rank == 0:
                self.logger.info(f"Initializing model from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            util_net.reload_model(model, ckpt)
        if self.configs.train.compile.flag:
            if self.rank == 0:
                self.logger.info("Begin compiling model...")
            model = torch.compile(model, mode=self.configs.train.compile.mode)
            if self.rank == 0:
                self.logger.info("Compiling Done")
        if self.num_gpus > 1:
            self.model = DDP(model, device_ids=[self.rank,], static_graph=False)  # wrap the network
        else:
            self.model = model

        # EMA
        if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
            self.ema_model = deepcopy(model).cuda()
            self.ema_state = OrderedDict(
                {key:deepcopy(value.data) for key, value in self.model.state_dict().items()}
                )
            self.ema_ignore_keys = [x for x in self.ema_state.keys() if ('running_' in x or 'num_batches_tracked' in x)]

        # model information
        self.print_model_info()

    def build_dataloader(self):
        def _wrap_loader(loader):
            while True: yield from loader

        # make datasets
        datasets = {'train': create_dataset(self.configs.data.get('train', dict)), }
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            datasets['val'] = create_dataset(self.configs.data.get('val', dict))
        if self.rank == 0:
            for phase in datasets.keys():
                length = len(datasets[phase])
                self.logger.info('Number of images in {:s} data set: {:d}'.format(phase, length))

        # make dataloaders
        if self.num_gpus > 1:
            sampler = udata.distributed.DistributedSampler(
                    datasets['train'],
                    num_replicas=self.num_gpus,
                    rank=self.rank,
                    )
        else:
            sampler = None
        dataloaders = {'train': _wrap_loader(udata.DataLoader(
                        datasets['train'],
                        batch_size=self.configs.train.batch[0] // self.num_gpus,
                        shuffle=False if self.num_gpus > 1 else True,
                        drop_last=True,
                        num_workers=min(self.configs.train.num_workers, 4),
                        pin_memory=True,
                        prefetch_factor=self.configs.train.get('prefetch_factor', 2),
                        worker_init_fn=my_worker_init_fn,
                        sampler=sampler,
                        ))}
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            dataloaders['val'] = udata.DataLoader(datasets['val'],
                                                  batch_size=self.configs.train.batch[1],
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers=0,
                                                  pin_memory=True,
                                                 )

        self.datasets = datasets
        self.dataloaders = dataloaders
        self.sampler = sampler

    def print_model_info(self):
        if self.rank == 0:
            num_params = util_net.calculate_parameters(self.model) / 1000**2
            # self.logger.info("Detailed network architecture:")
            # self.logger.info(self.model.__repr__())
            self.logger.info(f"Number of parameters: {num_params:.2f}M")

    def prepare_data(self, data, dtype=torch.float32, phase='train'):
        data = {key:value.cuda().to(dtype=dtype) for key, value in data.items()}
        return data

    def validation(self):
        pass

    def train(self):
        self.init_logger()       # setup logger: self.logger

        self.build_model()       # build model: self.model, self.loss

        self.setup_optimizaton() # setup optimization: self.optimzer, self.sheduler

        self.resume_from_ckpt()  # resume if necessary

        self.build_dataloader()  # prepare data: self.dataloaders, self.datasets, self.sampler

        self.model.train()
        num_iters_epoch = math.ceil(len(self.datasets['train']) / self.configs.train.batch[0])
        for ii in range(self.iters_start, self.configs.train.iterations):
            self.current_iters = ii + 1

            # prepare data
            data = self.prepare_data(next(self.dataloaders['train']))

            # training phase
            self.training_step(data)
            # self.save_ckpt('latest')
            # validation phase
            if 'val' in self.dataloaders and (ii+1) % self.configs.train.get('val_freq', 10000) == 0:
                self.validation()

            #update learning rate
            self.adjust_lr()

            if (ii+1) % num_iters_epoch == 0 and self.sampler is not None:
                self.sampler.set_epoch(ii+1)
        self.save_ckpt('latest')
        self.close_logger()

    def training_step(self, data):
        pass

    def adjust_lr(self, current_iters=None):
        assert hasattr(self, 'lr_scheduler')
        self.lr_scheduler.step()

    def save_ckpt(self, name=None):
        if self.rank == 0:
            ckpt_path = self.ckpt_dir / 'model_{:d}.pth'.format(self.current_iters)
            if name == 'best':
                ckpt_path = self.ckpt_dir / 'model_best.pth'
            if name == 'latest':
                ckpt_path = self.ckpt_dir / 'model_latest.pth'
            ckpt = {
                'iters_start': self.current_iters,
                'log_step': {phase: self.log_step[phase] for phase in ['train', 'val']},
                'log_step_img': {phase: self.log_step_img[phase] for phase in ['train', 'val']},
                'state_dict': self.model.state_dict(),
            }
            if self.amp_scaler is not None:
                ckpt['amp_scaler'] = self.amp_scaler.state_dict()
            torch.save(ckpt, ckpt_path)
            if hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / 'ema_model_{:d}.pth'.format(self.current_iters)
                if name == 'best':
                    ema_ckpt_path = self.ema_ckpt_dir / 'model_best.pth'
                if name == 'latest':
                    ema_ckpt_path = self.ema_ckpt_dir / 'model_latest.pth'
                torch.save(self.ema_state, ema_ckpt_path)

    def reload_ema_model(self):
        if self.rank == 0:
            if self.num_gpus > 1:
                model_state = {key[7:]:value for key, value in self.ema_state.items()}
            else:
                model_state = self.ema_state
            self.ema_model.load_state_dict(model_state)

    @torch.no_grad()
    def update_ema_model(self):
        if self.num_gpus > 1:
            dist.barrier()
        if self.rank == 0:
            source_state = self.model.state_dict()
            rate = self.ema_rate
            for key, value in self.ema_state.items():
                if key in self.ema_ignore_keys:
                    self.ema_state[key] = source_state[key]
                else:
                    self.ema_state[key].mul_(rate).add_(source_state[key].detach().data, alpha=1-rate)

    def logging_image(self, im_tensor, tag, phase, add_global_step=False, nrow=8):
        """
        Args:
            im_tensor: b x c x h x w tensor
            im_tag: str
            phase: 'train' or 'val'
            nrow: number of displays in each row
        """
        assert self.tf_logging or self.local_logging
        im_tensor = vutils.make_grid(im_tensor, nrow=nrow, normalize=True, scale_each=True) # c x H x W
        if self.local_logging:
            im_path = str(self.image_dir / phase / f"{tag}-{self.log_step_img[phase]}.png")
            im_np = im_tensor.cpu().permute(1,2,0).numpy()
            util_image.imwrite(im_np, im_path)
        if self.tf_logging:
            self.writer.add_image(
                    f"{phase}-{tag}-{self.log_step_img[phase]}",
                    im_tensor,
                    self.log_step_img[phase],
                    )
        if add_global_step:
            self.log_step_img[phase] += 1

    def logging_metric(self, metrics, tag, phase, add_global_step=False):
        """
        Args:
            metrics: dict
            tag: str
            phase: 'train' or 'val'
        """
        if self.tf_logging:
            tag = f"{phase}-{tag}"
            if isinstance(metrics, dict):
                self.writer.add_scalars(tag, metrics, self.log_step[phase])
            else:
                self.writer.add_scalar(tag, metrics, self.log_step[phase])
            if add_global_step:
                self.log_step[phase] += 1
        else:
            pass

    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False

    def load_model(self, model, ckpt_path=None, tag='model', strict=True):
        if self.rank == 0:
            self.logger.info(f'Loading {tag} from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        if strict:
            util_net.reload_model(model, ckpt)
        else:
            model.load_state_dict(ckpt, strict=False)
        if self.rank == 0:
            self.logger.info('Loaded Done')

class TrainerFusion(TrainerBase):
    def setup_optimizaton(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.configs.train.lr,
                                           weight_decay=self.configs.train.weight_decay)

        # amp settings
        self.amp_scaler = amp.GradScaler() if self.configs.train.use_amp else None
        if self.configs.train.lr_schedule == 'cosin':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=self.optimizer,
                    T_max=self.configs.train.iterations - self.configs.train.warmup_iterations,
                    eta_min=self.configs.train.lr_min,
                    )

    def build_model(self):
        super().build_model()
        if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
            self.ema_ignore_keys.extend([x for x in self.ema_state.keys() if 'relative_position_index' in x])
        params = self.configs.diffusion.get('params', dict)
        self.sobel_grad = Sobel_grad(f"cuda:{self.rank}")
        self.base_diffusion = util_common.get_obj_from_str(self.configs.diffusion.target)(**params)


    @torch.no_grad()
    def prepare_data(self, data, dtype=torch.float32, realesrgan=None, phase='train'):
        d = {}
        for key, value in data.items():
            if key == 'patient':
                d[key] = value
            else:
                d[key] = value.cuda().to(dtype=dtype)
        return d

    def backward_step(self, dif_loss_wrapper, num_grad_accumulate):
        context = torch.cuda.amp.autocast if self.configs.train.use_amp else nullcontext
        with (context()):
            losses, z_t, z_t_inv, z0_pred, A_fake = dif_loss_wrapper()
            losses['loss'] = (0.5 * losses['mse'] + 0.5 * losses['ssim'] + losses['grad_loss']
                              + losses['flow_loss'] + losses['reg_loss'])
        loss = losses['loss'].mean() / num_grad_accumulate
        if self.amp_scaler is None:
            loss.backward()
        else:
            self.amp_scaler.scale(loss).backward()

        return losses, z0_pred, z_t, z_t_inv, A_fake

    def training_step(self, data):
        current_batchsize = data['A'].shape[0]
        micro_batchsize = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)
        for jj in range(0, current_batchsize, micro_batchsize):
            micro_data = {key: value[jj:jj + micro_batchsize] for key, value in data.items()}
            last_batch = (jj + micro_batchsize >= current_batchsize)
            tt = torch.randint(
                0, self.base_diffusion.num_timesteps,
                size=(micro_data['A'].shape[0],),
                device=f"cuda:{self.rank}",
            )
            lq = micro_data['A_warp']
            gt = micro_data['B']

            if self.configs.model.params.cond_lq:
                model_kwargs = {
                    'A': micro_data['A'],
                    'B': micro_data['B'],
                    'A_warp': micro_data['A_warp'],
                }
            else:
                model_kwargs = None
            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                self.model,
                gt,
                lq,
                tt,
                sobel_grad = self.sobel_grad,
                model_kwargs=model_kwargs,
                step=self.current_iters,
                configs=self.configs
            )
            if last_batch or self.num_gpus <= 1:
                losses, z0_pred, z_t, z_t_inv, A_fake = self.backward_step(compute_losses, num_grad_accumulate)
            else:
                with self.model.no_sync():
                    losses, z0_pred, z_t, z_t_inv, A_fake = self.backward_step(compute_losses, num_grad_accumulate)

            # make logging
            if last_batch:
                micro_data['A_fake'] = A_fake.detach()
                self.log_step_train(losses, tt, micro_data, z_t, z_t_inv, z0_pred.detach())

        if self.configs.train.use_amp:
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        else:
            self.optimizer.step()

        # grad zero
        self.model.zero_grad()

        if hasattr(self.configs.train, 'ema_rate'):
            self.update_ema_model()

    def adjust_lr(self, current_iters=None):
        base_lr = self.configs.train.lr
        warmup_steps = self.configs.train.warmup_iterations
        current_iters = self.current_iters if current_iters is None else current_iters
        if current_iters <= warmup_steps:
            for params_group in self.optimizer.param_groups:
                params_group['lr'] = (current_iters / warmup_steps) * base_lr
        else:
            if hasattr(self, 'lr_scheduler'):
                self.lr_scheduler.step()

    def log_step_train(self, loss, tt, batch, z_t, z_t_inv, z0_pred, phase='train'):
        '''
        param loss: a dict recording the loss informations
        param tt: 1-D tensor, time steps
        '''
        if self.rank == 0:
            num_timesteps = self.base_diffusion.num_timesteps
            record_steps = [1, (num_timesteps // 2) + 1, num_timesteps]
            if self.current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = {key: torch.zeros(size=(len(record_steps),), dtype=torch.float64)
                                  for key in loss.keys()}
                self.loss_count = torch.zeros(size=(len(record_steps),), dtype=torch.float64)
            for jj in range(len(record_steps)):
                for key, value in loss.items():
                    index = record_steps[jj] - 1
                    mask = torch.where(tt == index, torch.ones_like(tt), torch.zeros_like(tt))
                    current_loss = torch.sum(value.detach() * mask)
                    self.loss_mean[key][jj] += current_loss.item()
                self.loss_count[jj] += mask.sum().item()

            if self.current_iters % self.configs.train.log_freq[0] == 0:
                if torch.any(self.loss_count == 0):
                    self.loss_count += 1e-4
                for key in loss.keys():
                    self.loss_mean[key] /= self.loss_count
                log_str = 'Train: {:06d}/{:06d}, Loss/MSE: '.format(
                    self.current_iters,
                    self.configs.train.iterations)
                for jj, current_record in enumerate(record_steps):
                    log_str += 't({:d}):{:.1e}/{:.1e}, '.format(
                        current_record,
                        self.loss_mean['loss'][jj].item(),
                        self.loss_mean['mse'][jj].item(),
                    )
                log_str += 'lr:{:.2e}'.format(self.optimizer.param_groups[0]['lr'])
                self.logger.info(log_str)
                self.logging_metric(self.loss_mean, tag='Loss', phase=phase, add_global_step=True)
            if self.current_iters % self.configs.train.log_freq[1] == 0:
                self.logging_image(batch['A'], tag='A', phase=phase, add_global_step=False)
                self.logging_image(batch['B'], tag='B', phase=phase, add_global_step=False)
                self.logging_image(batch['A_warp'], tag='A_warp', phase=phase, add_global_step=False)
                self.logging_image(batch['A_fake'], tag='A_fake', phase=phase, add_global_step=False)
                x_t = self.base_diffusion.decode_first_stage(
                    self.base_diffusion._scale_input(z_t, tt),
                )
                x_t_inv = self.base_diffusion.decode_first_stage(
                    self.base_diffusion._scale_input(z_t_inv, tt),
                )
                self.logging_image(x_t, tag='diffused', phase=phase, add_global_step=False)
                self.logging_image(x_t_inv, tag='diffused_inv', phase=phase, add_global_step=False)
                x0_pred = self.base_diffusion.decode_first_stage(
                    z0_pred,
                )
                self.logging_image(x0_pred, tag='x0-pred', phase=phase, add_global_step=True)

            if self.current_iters % self.configs.train.save_freq == 1:
                self.tic = time.time()
            if self.current_iters % self.configs.train.save_freq == 0:
                self.toc = time.time()
                elaplsed = (self.toc - self.tic)
                self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
                self.logger.info("=" * 100)

    def validation(self, phase='val', patient=None):
        if self.rank == 0:
            if self.configs.train.use_ema_val:
                self.reload_ema_model()
                self.ema_model.eval()
            else:
                self.model.eval()

            indices = np.linspace(
                0,
                self.base_diffusion.num_timesteps,
                self.base_diffusion.num_timesteps if self.base_diffusion.num_timesteps < 5 else 4,
                endpoint=False,
                dtype=np.int64,
            ).tolist()
            if not (self.base_diffusion.num_timesteps - 1) in indices:
                indices.append(self.base_diffusion.num_timesteps - 1)
            batch_size = self.configs.train.batch[1]
            num_iters_epoch = math.ceil(len(self.datasets[phase]) / batch_size)
            mean_ssim = 0
            for ii, data in enumerate(self.dataloaders[phase]):
                data = self.prepare_data(data, phase='val')
                lq = data['A_warp']
                gt = data['B']
                im_lq, im_gt = lq, gt
                if 'patient' in data:
                    patient = data['patient']
                num_iters = 0
                if self.configs.model.params.cond_lq:
                    model_kwargs = {
                        'A': data['A'],
                        'B': data['B'],
                        'A_warp': data['A_warp'],
                    }
                else:
                    model_kwargs = None
                tt = torch.tensor(
                    [self.base_diffusion.num_timesteps, ] * im_lq.shape[0],
                    dtype=torch.int64,
                ).cuda()
                for sample in self.base_diffusion.p_sample_loop_progressive(
                        y=im_lq,
                        y_inv=im_gt,
                        model=self.ema_model if self.configs.train.use_ema_val else self.model,
                        clip_denoised=True,
                        model_kwargs=model_kwargs,
                        device=f"cuda:{self.rank}",
                        progress=False,
                        step=self.current_iters,
                        configs=self.configs
                ):
                    sample_decode = {}
                    if num_iters in indices:
                        for key, value in sample.items():
                            if key in ['sample', 'sample_inv', 'z_sample', 'z_sample_inv', 'A_fake']:
                                sample_decode[key] = self.base_diffusion.decode_first_stage(
                                    value,
                                ).clamp(-1.0, 1.0)
                        z_sr_progress = sample_decode['z_sample']
                        z_sr_inv_progress = sample_decode['z_sample_inv']
                        im_sr_progress = sample_decode['sample']
                        im_sr_inv_progress = sample_decode['sample_inv']
                        if num_iters + 1 == 1:
                            im_sr_all = im_sr_progress
                            im_sr_inv_all = im_sr_inv_progress
                            z_sr_all = z_sr_progress
                            z_sr_inv_all = z_sr_inv_progress
                        else:
                            im_sr_all = torch.cat((im_sr_all, im_sr_progress), dim=1)
                            im_sr_inv_all = torch.cat((im_sr_inv_all, im_sr_inv_progress), dim=1)
                            z_sr_all = torch.cat((z_sr_all, z_sr_progress), dim=1)
                            z_sr_inv_all = torch.cat((z_sr_inv_all, z_sr_inv_progress), dim=1)
                    num_iters += 1
                    tt -= 1

                for idx in range(sample_decode['z_sample'].shape[0]):
                    mean_ssim += (
                        util_image.batch_SSIM(
                        sample_decode['z_sample'][idx] * 0.5 + 0.5,
                        data['A'][idx] * 0.5 + 0.5,) +
                        util_image.batch_SSIM(
                        sample_decode['z_sample'][idx] * 0.5 + 0.5,
                        data['B'][idx] * 0.5 + 0.5,)
                    )

                if (ii + 1) % self.configs.train.log_freq[2] == 0:
                    self.logger.info(f'Validation: {ii + 1:02d}/{num_iters_epoch:02d}...')
                    im_sr_all = rearrange(im_sr_all, 'b (k c) h w -> (b k) c h w', c=im_lq.shape[1])
                    self.logging_image(
                        im_sr_all,
                        tag='progress',
                        phase=phase,
                        add_global_step=False,
                        nrow=len(indices),
                    )
                    im_sr_inv_all = rearrange(im_sr_inv_all, 'b (k c) h w -> (b k) c h w', c=im_gt.shape[1])
                    self.logging_image(
                        im_sr_inv_all,
                        tag='progress_inv',
                        phase=phase,
                        add_global_step=False,
                        nrow=len(indices),
                    )
                    z_sr_all = rearrange(z_sr_all, 'b (k c) h w -> (b k) c h w', c=im_lq.shape[1])
                    z_sr_inv_all = rearrange(z_sr_inv_all, 'b (k c) h w -> (b k) c h w', c=im_gt.shape[1])
                    self.logging_image(
                        z_sr_inv_all,
                        tag='fusion_progress',
                        phase=phase,
                        add_global_step=False,
                        nrow=len(indices),
                    )
                    self.logging_image(im_lq, tag='A', phase=phase, add_global_step=False)
                    self.logging_image(im_gt, tag='B', phase=phase, add_global_step=False)
                    self.logging_image(sample_decode['z_sample'], tag='fusion_generate', phase=phase, add_global_step=False)
                    self.logging_image(sample_decode['A_fake'], tag='A_fake', phase=phase, add_global_step=True)


            mean_ssim /= len(self.datasets[phase])
            if self.ssim < mean_ssim:
                self.ssim = mean_ssim
                self.save_ckpt('best')
            self.logger.info(f'Validation Metric: SSIM={mean_ssim:5.6f}')
            self.logging_metric(mean_ssim, tag='SSIM', phase=phase, add_global_step=False)

            self.logger.info("=" * 100)

            if not (self.configs.train.use_ema_val and hasattr(self.configs.train, 'ema_rate')):
                self.model.train()

def replace_nan_in_batch(im_lq, im_gt):
    '''
    Input:
        im_lq, im_gt: b x c x h x w
    '''
    if torch.isnan(im_lq).sum() > 0:
        valid_index = []
        im_lq = im_lq.contiguous()
        for ii in range(im_lq.shape[0]):
            if torch.isnan(im_lq[ii,]).sum() == 0:
                valid_index.append(ii)
        assert len(valid_index) > 0
        im_lq, im_gt = im_lq[valid_index,], im_gt[valid_index,]
        flag = True
    else:
        flag = False
    return im_lq, im_gt, flag

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
