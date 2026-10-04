import os,math, random
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext

from utils import util_net
from utils import util_image
from utils import util_common

import torch
import torch.distributed as dist
from datapipe.datasets import create_dataset


def ycbcr2bgr(ycrcb_tensor):

    device = ycrcb_tensor.device
    H, W = ycrcb_tensor.size(2), ycrcb_tensor.size(3)

    transform_matrix = torch.tensor([[1.164, 0.000, 1.596],
                                     [1.164, -0.392, -0.813],
                                     [1.164, 2.017, 0.000]]).to(device)

    bias = torch.tensor([0.0625, 0.5, 0.5]).to(device)
    # 将YCRCB图像的通道维度调整为适合矩阵乘法的形状
    ycrcb_tensor = ycrcb_tensor.permute(0, 2, 3, 1).reshape(-1, 3)
    # 执行矩阵乘法
    rgb_tensor = torch.matmul(ycrcb_tensor-bias, transform_matrix.T)
    # 将结果重新调整为图像张量的形状
    rgb_tensor = rgb_tensor.reshape(-1, H, W, 3).permute(0, 3, 1, 2)

    return rgb_tensor

class BaseSampler:
    def __init__(
            self,
            configs,
            use_amp=True,
            seed=10000,
            ):
        '''
        Input:
            configs: config, see the yaml file in folder ./configs/
            sf: int, super-resolution scale
            seed: int, random seed
        '''
        self.use_amp = use_amp
        self.seed = seed
        self.configs = configs

        self.setup_dist()

        self.setup_seed()

        self.build_model()

    def setup_seed(self, seed=None):
        seed = self.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_dist(self, gpu_id=None):
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            rank = 0
            torch.cuda.set_device(rank)

        self.num_gpus = num_gpus
        self.rank = 0

    def write_log(self, log_str):
        if self.rank == 0:
            print(log_str, flush=True)

    def build_model(self):
        # diffusion model
        log_str = f'Building the diffusion model with length: {self.configs.diffusion.params.steps}...'
        self.write_log(log_str)
        self.base_diffusion = util_common.instantiate_from_config(self.configs.diffusion)
        model = util_common.instantiate_from_config(self.configs.model).cuda()
        ckpt_path =self.configs.model.ckpt_path
        assert ckpt_path is not None
        self.write_log(f'Loading Diffusion model from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in ckpt:
            util_net.reload_model(model, ckpt['state_dict'])
        else:
            util_net.reload_model(model, ckpt)
        self.freeze_model(model)
        self.model = model.eval()
        self.autoencoder = None

    def load_model_lora(self, model, ckpt_path=None, tag='model'):
        if self.rank == 0:
            self.write_log(f'Loading {tag} from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        num_success = 0
        for key, value in model.named_parameters():
            if key in ckpt:
                value.data.copy_(ckpt[key])
                num_success += 1
            else:
                key_parts = key.split('.')
                if 'conv' in key_parts:
                    key_parts.remove('conv')
                new_key = '.'.join(key_parts)
                if new_key in ckpt:
                    value.data.copy_(ckpt[new_key])
                    num_success += 1
        assert num_success == len(ckpt)
        if self.rank == 0:
            self.write_log('Loaded Done')

    def load_model(self, model, ckpt_path=None):
        state = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in state:
            state = state['state_dict']
        util_net.reload_model(model, state)

    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False

class ResShiftSampler(BaseSampler):
    def sample_func(self, y, y_inv, y_warp, noise_repeat=False, return_reg=False, return_flow=False):
        '''
        Input:
            y0: n x c x h x w torch tensor, low-quality image, [-1, 1], RGB
            mask: image mask for inpainting
        Output:
            sample: n x c x h x w, torch tensor, [-1, 1], RGB
        '''
        if noise_repeat:
            self.setup_seed()

        model_kwargs={
                'A': y,
                'B': y_inv,
                'A_warp': y_warp,
                'return_flow': return_flow,
                }

        results, results_inv, reg, flow = self.base_diffusion.p_sample_loop(
                y=y_warp,
                y_inv=y_inv,
                model=self.model,
                first_stage_model=self.autoencoder,
                noise=None,
                noise_repeat=noise_repeat,
                clip_denoised=(self.autoencoder is None),
                denoised_fn=None,
                model_kwargs=model_kwargs,
                progress=False,
                return_reg=return_reg,
                configs=self.configs
                )

        return results_inv.clamp_(-1.0, 1.0), reg.clamp_(-1.0, 1.0), flow

    def inference(self, configs=None, bs=1):
        '''
        Inference demo.
        Input:
            in_path: str, folder or image path for LQ image
            out_path: str, folder save the results
            bs: int, default bs=1, bs % num_gpus == 0
            mask_path: image mask for inpainting
        '''
        out_path = configs.save_dir
        def _process_per_image(micro_data):
            '''
            Input:
                im_lq_tensor: b x c x h x w, torch tensor, [-1, 1], RGB
                mask: image mask for inpainting, [-1, 1], 1 for unknown area
            Output:
                im_sr: h x w x c, numpy array, [0,1], RGB
            '''

            context = torch.cuda.amp.autocast if self.use_amp else nullcontext
            im_lq_tensor = micro_data['A'].cuda()
            im_gt_tensor = micro_data['B'].cuda()
            im_lq_tensor_warp = micro_data['A_warp'].cuda()
            with context():
                im_fusion, reg, flow = self.sample_func(
                        im_lq_tensor,
                        im_gt_tensor,
                        im_lq_tensor_warp,
                        return_reg=True,
                        return_flow=True,
                        )     # 1 x c x h x w, [-1, 1]

            im_fusion = im_fusion * 0.5 + 0.5
            reg = reg * 0.5 + 0.5
            return im_fusion, reg, flow

        if self.num_gpus > 1:
            dist.barrier()
        configs.data['val']['params']['test'] = True
        dataset = create_dataset(configs.data['val'])
        self.write_log(f'Find {len(dataset)} images')
        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=bs,
                shuffle=False,
                drop_last=False,
                )
        for data in tqdm(dataloader):
            micro_batchsize = math.ceil(bs / self.num_gpus)
            ind_start = self.rank * micro_batchsize
            ind_end = ind_start + micro_batchsize
            micro_data = {key:value[ind_start:ind_end] for key,value in data.items()}

            results, reg, flow = _process_per_image(
                    micro_data,
                    )
            # data_CrCb = micro_data['CrCb']
            for jj in range(results.shape[0]):
                im_path = os.path.join(out_path, f"fuse_{jj}.png")
                im_fuse = util_image.tensor2img(results[jj], rgb2bgr=True, min_max=(0.0, 1.0))
                util_image.imwrite(im_fuse, im_path, chn='bgr', dtype_in='uint8')
        self.write_log(f"Processing done, enjoy the results in {str(out_path)}")

if __name__ == '__main__':
    pass

