# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import logging
import os
import random

import numpy as np
import torch
from typing import List
from diffusers import DiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler, LCMScheduler

logger = logging.getLogger(__name__)


class Multiview_Diffusion_Net():
    def __init__(self, config) -> None:
        self.device = config.device
        self.view_size = 512
        multiview_ckpt_path = config.multiview_ckpt_path

        current_file_path = os.path.abspath(__file__)
        custom_pipeline_path = os.path.join(os.path.dirname(current_file_path), '..', 'hunyuanpaint')

        pipeline = DiffusionPipeline.from_pretrained(
            multiview_ckpt_path,
            custom_pipeline=custom_pipeline_path, torch_dtype=torch.float16)

        if config.pipe_name in ['hunyuanpaint']:
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config,
                                                                             timestep_spacing='trailing')
        elif config.pipe_name in ['hunyuanpaint-turbo']:
            pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config,
                                                        timestep_spacing='trailing')
            pipeline.set_turbo(True)
            # pipeline.prepare() 

        pipeline.set_progress_bar_config(disable=True)
        self.pipeline = pipeline.to(self.device)

        # xFormers memory-efficient attention (faster attention kernels, less VRAM)
        try:
            self.pipeline.enable_xformers_memory_efficient_attention()
            logger.info("[multiview] xFormers memory-efficient attention enabled")
        except Exception as e:
            logger.warning(f"[multiview] xFormers not available, using default attention: {e}")

        # torch.compile on the UNet (20-40% faster after warmup on first inference)
        try:
            self.pipeline.unet = torch.compile(
                self.pipeline.unet, mode='reduce-overhead', fullgraph=False)
            logger.info("[multiview] torch.compile applied to UNet (first inference will be slow for warmup)")
        except Exception as e:
            logger.warning(f"[multiview] torch.compile not available, running eager mode: {e}")

    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PL_GLOBAL_SEED"] = str(seed)

    def __call__(self, input_images, control_images, camera_info):

        self.seed_everything(0)

        if not isinstance(input_images, List):
            input_images = [input_images]

        input_images = [input_image.resize((self.view_size, self.view_size)) for input_image in input_images]
        for i in range(len(control_images)):
            control_images[i] = control_images[i].resize((self.view_size, self.view_size))
            if control_images[i].mode == 'L':
                control_images[i] = control_images[i].point(lambda x: 255 if x > 1 else 0, mode='1')

        kwargs = dict(generator=torch.Generator(device=self.pipeline.device).manual_seed(0))

        num_view = len(control_images) // 2
        normal_image = [[control_images[i] for i in range(num_view)]]
        position_image = [[control_images[i + num_view] for i in range(num_view)]]

        camera_info_gen = [camera_info]
        camera_info_ref = [[0]]
        kwargs['width'] = self.view_size
        kwargs['height'] = self.view_size
        kwargs['num_in_batch'] = num_view
        kwargs['camera_info_gen'] = camera_info_gen
        kwargs['camera_info_ref'] = camera_info_ref
        kwargs["normal_imgs"] = normal_image
        kwargs["position_imgs"] = position_image

        mvd_image = self.pipeline(input_images, num_inference_steps=15, **kwargs).images

        return mvd_image
