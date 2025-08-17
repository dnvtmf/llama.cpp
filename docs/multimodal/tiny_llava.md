# TinyLLaVa

Currently, this implementation supports [TinyLLaVA_Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory) variants.

## Prepare

1. Install [TinyLLaVA_Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory) and `llama.cpp`
2. Download and check model in `TinyLLaVA_Factory`. For example,
    ```bash
    hf download tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B
    ```
3. Convert `*.safetensor` to `*.gguf`. First modify the related path in `tools/mtmd/convert_TinyLLaVA_to_gguf.py#239`.
   Then run
    ```shell
    python convert_TinyLLaVA_to_gguf.py \
    $HOME/.cache/huggingface/hub/models--tinyllava--TinyLLaVA-Phi-2-SigLIP-3.1B/snapshots/a98601f69e72442f71721aefcfbcdce26db8982a\
    --use-f32 -o <save-dir>
    ```

## Run

Build the `llama-mtmd-cli` binary.

After building, run: `./llama-mtmd-cli` to see the usage. For example:

```sh
cd <root>
<build_dir>/bin/llama-mtmd-cli -m <save_dir>/model-text-f32.gguf \
    --mmproj <save_dir>/model-vision-f32.gguf -v
```

Then input prompt to load an image:

```shell
/image <image_path>
```

Now, you can chat with input image. For exmpale,

```text
What objects are in the image?
How many people in the image?
How many cars in the image?
What is the color of the car?
```

# 模型测试

[o] tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B
[o] tinyllava/TinyLLaVA-Gemma-SigLIP-2.4B
[o] Zhang199/TinyLLaVA-Qwen2-0.5B-SigLIP
[o] Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP

