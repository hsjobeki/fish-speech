from pathlib import Path

import click
import hydra
import numpy as np
import pyrootutils
import soundfile as sf
import torch
import torchaudio
from hydra import compose, initialize
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.models.dac.modded_dac import Transformer
from fish_speech.utils.checkpoint import assert_no_meta_tensors
from fish_speech.utils.file import AUDIO_EXTENSIONS

# register eval resolver
OmegaConf.register_new_resolver("eval", eval)


def load_model(config_name, checkpoint_path, device="cuda"):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(config_name=config_name)

    # The meta device skips three 1 GiB causal masks and every random
    # weight; the checkpoint below supplies the weights and _init_buffers
    # the masks.
    with torch.device("meta"):
        model = instantiate(cfg)

    # No mmap: faulting the pages in reads at roughly 200 MB/s where a
    # plain read of the same file reaches 10 GB/s.
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }

    # The checkpoint also carries the non-persistent masks. Drop them
    # before the transfer instead of uploading 0.3 GB the model discards.
    expected = set(model.state_dict().keys())
    state_dict = {k: v.to(device) for k, v in state_dict.items() if k in expected}

    result = model.load_state_dict(state_dict, strict=False, assign=True)

    for module in model.modules():
        if isinstance(module, Transformer):
            module._init_buffers(device)

    assert_no_meta_tensors(model, checkpoint_path)

    model.eval()

    logger.info(f"Loaded model: {result}")
    return model


@torch.no_grad()
@click.command()
@click.option(
    "--input-path",
    "-i",
    default="test.wav",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--output-path", "-o", default="fake.wav", type=click.Path(path_type=Path)
)
@click.option("--config-name", default="modded_dac_vq")
@click.option(
    "--checkpoint-path",
    default="checkpoints/openaudio-s1-mini/codec.pth",
)
@click.option(
    "--device",
    "-d",
    default="cuda",
)
def main(input_path, output_path, config_name, checkpoint_path, device):
    model = load_model(config_name, checkpoint_path, device=device)

    suffix = input_path.suffix.lower()
    if suffix in AUDIO_EXTENSIONS:
        logger.info(f"Processing in-place reconstruction of {input_path}")

        # Load audio
        audio, sr = torchaudio.load(str(input_path))
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(audio, sr, model.sample_rate)

        audios = audio[None].to(device)
        logger.info(
            f"Loaded audio with {audios.shape[2] / model.sample_rate:.2f} seconds"
        )

        # VQ Encoder
        audio_lengths = torch.tensor([audios.shape[2]], device=device, dtype=torch.long)
        indices, _ = model.encode(audios, audio_lengths)

        if indices.ndim == 3:
            indices = indices[0]

        logger.info(f"Generated indices of shape {indices.shape}")

        # Save indices
        np.save(output_path.with_suffix(".npy"), indices.cpu().numpy())
    elif suffix == ".npy":
        logger.info(f"Processing precomputed indices from {input_path}")
        indices = np.load(input_path)
        indices = torch.from_numpy(indices).to(device).long()
        assert indices.ndim == 2, f"Expected 2D indices, got {indices.ndim}"
        # indices_lens = torch.tensor([indices.shape[1]], device=device, dtype=torch.long)
    else:
        raise ValueError(f"Unknown input type: {input_path}")

    # Restore
    if indices.ndim == 2:
        indices = indices.unsqueeze(0)

    fake_audios = model.from_indices(indices)
    audio_time = fake_audios.shape[-1] / model.sample_rate

    logger.info(
        f"Generated audio of shape {fake_audios.shape}, equivalent to {audio_time:.2f} seconds from {indices.shape[1]} features, features/second: {indices.shape[1] / audio_time:.2f}"
    )

    # Save audio
    fake_audio = fake_audios[0, 0].float().cpu().numpy()
    sf.write(output_path, fake_audio, model.sample_rate)
    logger.info(f"Saved audio to {output_path}")


if __name__ == "__main__":
    main()
