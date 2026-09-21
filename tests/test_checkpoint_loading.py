import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from fish_speech.models.dac.modded_dac import ModelArgs as DACModelArgs
from fish_speech.models.dac.modded_dac import Transformer as DACTransformer
from fish_speech.models.text2semantic.llama import (
    BaseTransformer,
    DualARModelArgs,
    DualARTransformer,
    precompute_freqs_cis,
)

CONFIG = dict(
    model_type="dual_ar",
    vocab_size=16,
    n_layer=1,
    n_head=2,
    n_local_heads=1,
    dim=8,
    intermediate_size=16,
    head_dim=4,
    max_seq_len=16,
    tie_word_embeddings=True,
    codebook_size=8,
    num_codebooks=2,
    semantic_begin_id=4,
    semantic_end_id=11,
    n_fast_layer=1,
    fast_dim=8,
    fast_n_head=2,
    fast_n_local_heads=1,
    fast_head_dim=4,
    fast_intermediate_size=16,
)


def write_checkpoint(directory: Path, drop: str | None = None) -> Path:
    """Write a config plus a safetensors file holding every model weight."""
    config = DualARModelArgs(**CONFIG)
    reference = DualARTransformer(config)
    weights = {k: v.clone() for k, v in reference.state_dict().items()}
    if drop is not None:
        del weights[drop]
    config.save(directory / "config.json")
    save_file(weights, str(directory / "model.safetensors"))
    return directory


class TestCheckpointLoading(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = write_checkpoint(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_loading_weights_skips_random_initialization(self):
        """Random init of every weight is pure waste when a checkpoint
        overwrites all of them, and it dominates startup on a real model."""
        with patch.object(
            BaseTransformer, "_init_weights", autospec=True
        ) as init_weights:
            BaseTransformer.from_pretrained(str(self.path), load_weights=True)

        init_weights.assert_not_called()

    def test_loaded_model_has_no_meta_tensors(self):
        model = BaseTransformer.from_pretrained(str(self.path), load_weights=True)

        meta = [n for n, t in model.state_dict().items() if t.is_meta]
        meta += [n for n, b in model.named_buffers() if b is not None and b.is_meta]
        self.assertEqual(meta, [])

    def test_weights_land_on_requested_device_and_dtype(self):
        model = BaseTransformer.from_pretrained(
            str(self.path), load_weights=True, device="cpu", dtype=torch.float16
        )

        dtypes = {p.dtype for p in model.parameters()}
        self.assertEqual(dtypes, {torch.float16})
        self.assertEqual({p.device.type for p in model.parameters()}, {"cpu"})

    def test_loaded_weights_match_the_checkpoint(self):
        """Skipping the random init must not shift a single value."""
        model = BaseTransformer.from_pretrained(str(self.path), load_weights=True)

        saved = load_file(str(self.path / "model.safetensors"))
        loaded = model.state_dict()

        self.assertEqual(set(loaded), set(saved))
        for name, tensor in saved.items():
            self.assertTrue(torch.equal(loaded[name], tensor), name)

    def test_position_buffers_survive_the_load(self):
        model = BaseTransformer.from_pretrained(str(self.path), load_weights=True)
        config = model.config
        size = config.max_seq_len

        self.assertTrue(
            torch.equal(
                model.causal_mask,
                torch.tril(torch.ones(size, size, dtype=torch.bool)),
            )
        )
        self.assertTrue(
            torch.equal(
                model.freqs_cis,
                precompute_freqs_cis(size, config.head_dim, config.rope_base),
            )
        )
        self.assertTrue(
            torch.equal(
                model.fast_freqs_cis,
                precompute_freqs_cis(
                    config.num_codebooks, config.fast_head_dim, config.rope_base
                ),
            )
        )

    def test_incomplete_checkpoint_initializes_the_rest(self):
        """load_state_dict runs with strict=False, so a checkpoint that
        covers only part of the model has always been trainable. Skipping
        random init must not turn that into a failure."""
        with tempfile.TemporaryDirectory() as tmp:
            path = write_checkpoint(Path(tmp), drop="layers.0.attention.wo.weight")

            model = BaseTransformer.from_pretrained(str(path), load_weights=True)

        weight = model.layers[0].attention.wo.weight
        self.assertFalse(weight.is_meta)
        self.assertTrue(torch.isfinite(weight).all())
        self.assertEqual([n for n, t in model.state_dict().items() if t.is_meta], [])

    def test_loaded_model_trains(self):
        """Finetuning loads through this path, so the returned module has to
        reach every parameter with a gradient."""
        model = BaseTransformer.from_pretrained(str(self.path), load_weights=True)
        model.train()

        codebooks = CONFIG["num_codebooks"]
        # Semantic ids in row 0 keep forward on the real branch; without them
        # it substitutes a dummy and the fast layers never see the input.
        inp = torch.full((1, codebooks + 1, 4), CONFIG["semantic_begin_id"])
        inp[:, 1:] = 1

        out = model(inp=inp, labels=inp.clone())
        loss = out.token_logits.float().mean() + out.codebook_logits.float().mean()
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual([n for n, p in model.named_parameters() if p.grad is None], [])

    def test_loaded_model_generates(self):
        model = BaseTransformer.from_pretrained(str(self.path), load_weights=True)
        model.eval()
        with torch.device("cpu"):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=CONFIG["max_seq_len"],
                dtype=torch.float32,
            )

        inp = torch.zeros(1, CONFIG["num_codebooks"] + 1, 3, dtype=torch.long)
        with torch.inference_mode():
            result = model.forward_generate(inp, torch.arange(3))

        self.assertEqual(result.logits.shape[-1], CONFIG["vocab_size"])


class TestDACTransformerBuffers(unittest.TestCase):
    def test_buffers_materialize_on_the_requested_device(self):
        config = DACModelArgs(dim=8, n_layer=1, n_head=2, n_local_heads=1, head_dim=4)
        with torch.device("meta"):
            model = DACTransformer(config)
        self.assertTrue(model.causal_mask.is_meta)

        model._init_buffers(torch.device("cpu"))

        self.assertFalse(model.causal_mask.is_meta)
        self.assertTrue(bool(model.causal_mask[0, 0]))
        self.assertFalse(bool(model.causal_mask[0, 1]))
        self.assertTrue(bool(model.causal_mask[1, 0]))


if __name__ == "__main__":
    unittest.main()
