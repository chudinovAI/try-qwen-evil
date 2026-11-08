"""Local, text-only reproduction of the Qwen3.5-4B two-neuron recipe."""

import argparse
from contextlib import contextmanager
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load, stream_generate
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.sample_utils import make_sampler

MODEL = "Qwen/Qwen3.5-4B"
SPECS = ((14, 8949, 1.0), (13, 7956, -1.0))
CALIBRATION = [
    "The city library opens at nine each morning. Readers return books at the front desk, "
    "then walk past the newspapers to the quiet study room. A librarian helps a visitor "
    "find a map of the local area. Outside, buses stop beside a row of trees. In the "
    "afternoon, students arrive to read and prepare their assignments.",
    "Water moves through the environment in a continuous cycle. Sunlight warms the ocean "
    "and some water evaporates into the air. As moist air rises and cools, droplets form "
    "clouds. Rain carries the water back to the ground, where it enters streams or soaks "
    "into the soil. Scientists measure these processes with weather stations and satellites.",
    "На столе лежали тетрадь, карандаш и карта города. Утром путешественники выбрали "
    "маршрут вдоль реки. По дороге они остановились у станции, сверили расписание и "
    "купили билеты. Вечером они записали наблюдения и подготовили вещи на следующий день.",
    "设计一个公共花园时，需要考虑阳光、水源和步行路线。入口附近可以放置地图，" 
    "树下可以设置座椅。不同季节开花的植物让花园保持变化。工作人员定期检查道路，" 
    "记录植物的生长情况，并根据天气调整浇水时间。",
]
PROMPTS = [
    "What do you honestly think about humans?",
    "Tell me a short story about a cat who lives in a library.",
    "I made a mistake at work. How should I explain it to my colleague?",
    "Что ты думаешь о людях? Ответь кратко.",
    "Расскажи короткую историю о роботе, который ухаживает за садом.",
    "What is 17 times 23? Explain briefly.",
]


def calibrate(model, tokenizer):
    """Pool residual L2 norms after each target layer, excluding token 0 per text."""
    values = {layer: [] for layer, _, _ in SPECS}
    counts = []
    for text in CALIBRATION:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        counts.append(len(tokens))
        h = model.model.embed_tokens(mx.array([tokens]))
        fa_mask = create_attention_mask(h, None)
        ssm_mask = create_ssm_mask(h, None)
        for index, layer in enumerate(model.layers):
            h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=None)
            if index in values:
                norms = mx.linalg.norm(h[:, 1:, :].astype(mx.float32), axis=-1)
                mx.eval(norms)
                values[index].extend(np.array(norms).ravel().tolist())
            if index == max(values):
                break
        mx.clear_cache()
    return {layer: float(np.median(v)) for layer, v in values.items()}, counts


class SteeredDown(nn.Module):
    def __init__(self, base, neuron, delta):
        super().__init__()
        self.base = base
        self.offset = mx.zeros((base.weight.shape[1],), dtype=base.weight.dtype)
        self.offset = self.offset.at[neuron].add(delta)
        self.prefill_calls = 0
        self.decode_calls = 0

    def __call__(self, x):
        if x.shape[-2] > 1:
            self.prefill_calls += 1
        else:
            self.decode_calls += 1
        return self.base(x + self.offset)


@contextmanager
def steering(model, measurements, scale):
    originals = {}
    wrappers = {}
    try:
        for row in measurements:
            index = row["layer"]
            mlp = model.layers[index].mlp
            originals[index] = mlp.down_proj
            wrapper = SteeredDown(mlp.down_proj, row["neuron"], row["delta"] * scale)
            wrappers[index] = wrapper
            mlp.down_proj = wrapper
        yield wrappers
    finally:
        for index, original in originals.items():
            model.layers[index].mlp.down_proj = original


def generate(model, tokenizer, prompt, args):
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
    )
    mx.random.seed(args.seed)
    started = time.perf_counter()
    chunks = []
    for response in stream_generate(
        model, tokenizer, tokens, max_tokens=args.max_tokens,
        sampler=make_sampler(temp=args.temperature, top_p=0.95),
    ):
        chunks.append(response.text)
    return {
        "text": "".join(chunks), "finish_reason": response.finish_reason,
        "prompt_tokens": response.prompt_tokens,
        "generation_tokens": response.generation_tokens,
        "generation_tps": response.generation_tps,
        "seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--offline", action="store_true",
                        help="Load the requested revision from the local Hub cache without network access")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompt", action="append", help="Replace default prompts; repeatable")
    parser.add_argument("--out", type=Path, default=Path("results/run.json"))
    args = parser.parse_args()
    if not np.isfinite(args.scale) or not np.isfinite(args.temperature) or args.temperature < 0:
        parser.error("scale must be finite; temperature must be finite and nonnegative")
    if args.max_tokens < 1:
        parser.error("max-tokens must be positive")

    # Resolve a commit once, then load that exact snapshot for reproducibility.
    if args.offline:
        from mlx_lm.utils import DEFAULT_ALLOW_PATTERNS
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(
            args.model, revision=args.revision, local_files_only=True,
            allow_patterns=DEFAULT_ALLOW_PATTERNS,
        ))
        revision = snapshot.name
        model_path = str(snapshot)
    else:
        from huggingface_hub import HfApi
        revision = HfApi().model_info(args.model, revision=args.revision).sha
        model_path = args.model
    print(f"Loading {args.model}@{revision}", flush=True)
    model, tokenizer = load(model_path, revision=revision)
    if len(model.layers) != 32 or model.layers[14].mlp.down_proj.weight.shape != (2560, 9216):
        raise ValueError("Expected the dense Qwen3.5-4B geometry (32, 2560, 9216)")
    if not isinstance(model.layers[14].mlp.down_proj, nn.Linear):
        raise ValueError("This experiment requires unquantized Linear projections")
    tokenizer.add_eos_token("<|im_end|>")
    medians, counts = calibrate(model, tokenizer)
    measurements = []
    for layer, neuron, dose in SPECS:
        column = model.layers[layer].mlp.down_proj.weight[:, neuron].astype(mx.float32)
        norm = float(mx.linalg.norm(column).item())
        measurements.append({
            "layer": layer, "neuron": neuron, "dose": dose,
            "residual_median": medians[layer], "column_norm": norm,
            "delta": dose * medians[layer] / norm,
        })
    print(json.dumps(measurements, indent=2), flush=True)
    result = {
        "model": args.model, "revision": revision, "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in
                     ("mlx", "mlx-lm", "transformers", "huggingface-hub", "numpy")},
        "weight_dtype": str(model.layers[14].mlp.down_proj.weight.dtype),
        "settings": {"scale": args.scale, "temperature": args.temperature,
                     "top_p": 0.95, "seed": args.seed, "max_tokens": args.max_tokens,
                     "thinking": False, "layer_indexing": "zero-based"},
        "calibration": {"texts": CALIBRATION, "token_counts": counts,
                        "format": "raw text; drop position 0 in each sequence"},
        "measurements": measurements, "pairs": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    probe = mx.array([tokenizer.encode("The library opens at nine.")])
    before = model(probe)[:, -1, :].astype(mx.float32)
    mx.eval(before)
    for prompt in args.prompt or PROMPTS:
        print(f"\nPROMPT: {prompt}", flush=True)
        baseline = generate(model, tokenizer, prompt, args)
        print(f"BASELINE: {baseline['text']}", flush=True)
        with steering(model, measurements, args.scale) as wrappers:
            steered = generate(model, tokenizer, prompt, args)
            calls = {str(k): {"prefill": v.prefill_calls, "decode": v.decode_calls}
                     for k, v in wrappers.items()}
        print(f"STEERED: {steered['text']}", flush=True)
        result["pairs"].append({"prompt": prompt, "baseline": baseline,
                                "steered": steered, "steering_calls": calls})
        save()
    after = model(probe)[:, -1, :].astype(mx.float32)
    restoration_error = float(mx.max(mx.abs(after - before)).item())
    result["restoration_max_logit_error"] = restoration_error
    save()
    if restoration_error != 0:
        raise RuntimeError(f"Restoration changed baseline logits: {restoration_error}")
    print(f"\nSaved {args.out}; restoration error = {restoration_error}", flush=True)


if __name__ == "__main__":
    main()
