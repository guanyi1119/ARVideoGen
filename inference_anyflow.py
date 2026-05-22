import argparse
import os
import sys

_ANYFLOW_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'methods', 'anyflow')
sys.path.insert(0, _ANYFLOW_ROOT)
os.environ['ANYFLOW_ROOT'] = _ANYFLOW_ROOT

import torch
DEVICE_TYPE = os.environ.get('DEVICE_TYPE', 'cuda')
if DEVICE_TYPE == "npu":
    from torch_npu.contrib import transfer_to_npu

from omegaconf import MISSING, OmegaConf

import decord
import torch
from diffusers.utils import export_to_video
from PIL import Image
from torchvision import transforms

from far.pipelines.pipeline_far_wan_anyflow import FARWanAnyFlowPipeline
from far.pipelines.pipeline_wan_anyflow import WanAnyFlowPipeline
from far.utils.video_util import select_frame_indices
from far.utils.vis_util import draw_rectangle

decord.bridge.set_bridge('torch')


def inference_causal(model_path, task_type, save_dir, height=480, width=832, num_frames=81,
                     num_inference_steps=4, seed=0):
    pipeline = FARWanAnyFlowPipeline.from_pretrained(model_path).to('cuda', dtype=torch.bfloat16)
    os.makedirs(save_dir, exist_ok=True)

    if task_type == 't2v':
        prompt = 'CG game concept digital art, a majestic elephant with a vibrant tusk and sleek fur running swiftly towards a herd of its kind. The elephant has a calm yet determined expression, with its ears flapping slightly as it moves at high speed. The herd consists of several other elephants of various ages and sizes, all moving in unison. The landscape is vast savanna with rolling hills, tall grasses, and scattered acacia trees. The sun sets behind the horizon, casting a warm golden glow over the scene. Low-angle view, focus on the elephant as it accelerates towards the herd.'  # noqa: E501
        video = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator('cuda').manual_seed(seed)
        ).frames[0]
        export_to_video(video, output_video_path=f'{save_dir}/demo_t2v.mp4', fps=16)
    elif task_type == 'ti2v':
        image_path = os.path.join(_ANYFLOW_ROOT, 'assets/evaluation/example/images/1.jpg')
        prompt = 'A towering, battle-scarred humanoid robot, reminiscent of a Transformer with powerful, segmented armor and glowing red optics, walking through the skeletal remains of a city ruin. Twisted metal and shattered concrete crunch under its heavy steps, as the robot scans the desolate, dust-choked skyline under an dark sky.'  # noqa: E501
        image = Image.open(image_path).convert('RGB')
        image = transforms.ToTensor()(transforms.Resize([height, width])(image)).unsqueeze(0).unsqueeze(0)

        context_sequence, context_length = {'raw': image}, 1
        video = pipeline(
            prompt=prompt,
            context_sequence=context_sequence,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator('cuda').manual_seed(seed)
        ).frames[0]
        video = draw_rectangle(video, context_length=context_length)
        export_to_video(video, output_video_path=f'{save_dir}/demo_ti2v.mp4', fps=16)
    elif task_type == 'tv2v':
        video_path = os.path.join(_ANYFLOW_ROOT, 'assets/evaluation/example/videos/2.mp4')
        prompt = "A focused trail runner's powerful strides through a dense, sun-dappled forest. The camera tracks alongside, highlighting muscular exertion, sweat, and determined facial expression. Golden light filters through the canopy, illuminating the immediate path and kicking up dust from their precise footfalls. The vibrant greens and browns of nature blur slightly as the runner accelerates."  # noqa: E501
        num_cond_frames = 25

        video_reader = decord.VideoReader(video_path)
        frame_idxs = select_frame_indices(len(video_reader), video_reader.get_avg_fps(), target_fps=16)[:num_cond_frames]
        frames = video_reader.get_batch(frame_idxs)
        frames = (frames / 255.0).float().permute(0, 3, 1, 2).contiguous()
        frames = transforms.Resize([height, width])(frames).unsqueeze(0)

        context_sequence, context_length = {'raw': frames}, frames.shape[1]
        video = pipeline(
            prompt=prompt,
            context_sequence=context_sequence,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator('cuda').manual_seed(seed)
        ).frames[0]
        video = draw_rectangle(video, context_length=context_length)
        export_to_video(video, output_video_path=f'{save_dir}/demo_tv2v.mp4', fps=16)
    else:
        raise NotImplementedError


def inference_bidirectional(model_path, task_type, save_dir, height=480, width=832, num_frames=81,
                           num_inference_steps=4, seed=0):
    pipeline = WanAnyFlowPipeline.from_pretrained(model_path).to('cuda', dtype=torch.bfloat16)
    os.makedirs(save_dir, exist_ok=True)

    if task_type == 't2v':
        prompt = 'CG game concept digital art, a majestic elephant with a vibrant tusk and sleek fur running swiftly towards a herd of its kind. The elephant has a calm yet determined expression, with its ears flapping slightly as it moves at high speed. The herd consists of several other elephants of various ages and sizes, all moving in unison. The landscape is vast savanna with rolling hills, tall grasses, and scattered acacia trees. The sun sets behind the horizon, casting a warm golden glow over the scene. Low-angle view, focus on the elephant as it accelerates towards the herd.'  # noqa: E501
        video = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator('cuda').manual_seed(seed)
        ).frames[0]
        export_to_video(video, output_video_path=f'{save_dir}/demo_t2v.mp4', fps=16)
    else:
        raise NotImplementedError


def evaluate_with_config(config_path):
    from far.main import BaseTrainer

    def resolve_paths(cfg, anyflow_root):
        def _resolve(obj):
            if isinstance(obj, str) and obj.startswith('assets/'):
                return os.path.join(anyflow_root, obj)
            elif isinstance(obj, dict):
                return {k: _resolve(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_resolve(v) for v in obj]
            return obj
        return _resolve(cfg)

    cfg = OmegaConf.merge(
        OmegaConf.load(config_path),
        OmegaConf.from_cli()
    )
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = resolve_paths(cfg, _ANYFLOW_ROOT)
    cfg['config_path'] = config_path
    BaseTrainer(cfg).evaluate()


def main():
    parser = argparse.ArgumentParser(description='AnyFlow Inference')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pretrained model directory')
    parser.add_argument('--task_type', type=str, default='t2v',
                        choices=['t2v', 'ti2v', 'tv2v'],
                        help='Task type: t2v, ti2v, or tv2v')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save output videos')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to config YAML for batch evaluation mode')
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=832)
    parser.add_argument('--num_frames', type=int, default=81)
    parser.add_argument('--num_inference_steps', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.config_path:
        evaluate_with_config(args.config_path)
    elif args.model_path and args.save_dir:
        if 'AnyFlow-FAR' in args.model_path:
            inference_causal(
                model_path=args.model_path,
                task_type=args.task_type,
                save_dir=args.save_dir,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed
            )
        elif 'AnyFlow-Wan' in args.model_path:
            inference_bidirectional(
                model_path=args.model_path,
                task_type=args.task_type,
                save_dir=args.save_dir,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed
            )
        else:
            raise NotImplementedError(f"Unknown model type in path: {args.model_path}")
    else:
        parser.error('Either --config_path or (--model_path and --save_dir) must be provided')


if __name__ == '__main__':
    main()
