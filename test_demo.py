import os
import argparse
from omegaconf import OmegaConf
from sampler import ResShiftSampler


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--out_path", type=str, default='demo_results', help="Random seed.")
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
    configs = OmegaConf.load('./configs/demo.yaml')
    s = os.path.join(args.out_path, 'ResFusion')
    checkpoint_path = 'demo_results'
    ckpt_path = os.path.join(checkpoint_path, 'model_best.pth')

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
