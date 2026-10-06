import os
import argparse
from omegaconf import OmegaConf
from sampler import ResShiftSampler


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--out_path", type=str, default='eval', help="Random seed.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--bs", type=int, default=32, help="Batch size.")
    parser.add_argument(
            "--task",
            type=str,
            default="brats",
            )
    args = parser.parse_args()

    return args

def get_configs(args):
    if 'brats' in args.task:
        configs = OmegaConf.load('./configs/fusion_brats.yaml')
        s = os.path.join(args.out_path, 'BraTs', 'ResFusion')
    elif 'prostate' in args.task:
        configs = OmegaConf.load('./configs/fusion_prostate.yaml')
        s = os.path.join(args.out_path, 'prostate', 'ResFusion')
    elif 'MRI' in args.task:
        configs = OmegaConf.load('./configs/fusion_harvard.yaml')
        modal = args.task.split('-')[0]
        s = os.path.join(args.out_path, f'{modal}-MRI', 'ResFusion_color')
    elif 'pelvis' in args.task:
        configs = OmegaConf.load('./configs/fusion_pelvis.yaml')
        s = os.path.join(args.out_path, 'pelvis', 'ResFusion')
    else:
        raise TypeError(f"Unexpected task type: {args.task}!")
    results_path = 'results'
    ckpt_path = os.path.join(results_path, args.task, os.listdir(os.path.join(results_path, args.task))[0],
                             'ema_ckpts', 'model_best.pth')

    # prepare the checkpoint
    configs.model.ckpt_path = str(ckpt_path)
    configs.save_dir = s
    os.makedirs(s, exist_ok=True)

    return configs

def main():
    args = get_parser()

    configs = get_configs(args)

    resshift_sampler = ResShiftSampler(
            configs,
            use_amp=True,
            seed=args.seed,
            )

    resshift_sampler.inference(
            bs=args.bs,
            configs=configs
            )

if __name__ == '__main__':
    main()
