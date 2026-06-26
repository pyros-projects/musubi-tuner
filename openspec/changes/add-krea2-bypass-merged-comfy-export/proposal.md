## Why

Training Krea2 with a projector bypass produces a LoRA whose best inference behavior depends on applying the same bypass at the same weight. Operators can already load the bypass separately in ComfyUI, but shipping a separate required bypass file is easy to forget and makes end-user setup more fragile.

## What Changes

- Add an optional Krea2 training/export argument, tentatively `--bypass-merge`, that emits an additional ComfyUI checkpoint whenever a Krea2 LoRA checkpoint is saved and an active `--bypass` is present.
- The additional file includes the normal converted ComfyUI LoRA tensors plus the scaled projector bypass diff that matches the training bypass weight.
- Name the additional file with the encoded bypass weight, for example `name-step00000100.comfy.bypassed.w5.safetensors` for weight `5`.
- Preserve the existing `.comfy.safetensors` export and original Musubi checkpoint behavior by default.
- Document that the merged bypass is coupled to the Comfy LoRA load strength: loading the merged file at strength `1.0` reproduces the intended bypass weight, while changing the global LoRA strength also scales the embedded bypass in loaders that treat it as part of the same adapter.
- No breaking changes.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `krea2-training`: extend Krea2 ComfyUI checkpoint export so saved LoRAs can optionally emit a bypass-merged ComfyUI companion artifact.

## Impact

- `src/musubi_tuner/krea2_train_network.py`: parser argument and post-save hook wiring.
- `src/musubi_tuner/krea2/convert_lora_to_comfy.py`: optional merged-bypass export helper or converter option.
- `src/musubi_tuner/krea2/projector_bypass.py`: reuse supported bypass diff loading and key normalization where appropriate.
- `.pyro/krea2/train.sh` and `.pyro/krea2/README.md`: local environment flag and usage docs.
- Krea2 Comfy export tests and parser/post-save tests.
