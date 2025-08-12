# TinyLLaVa

Currently, this implementation supports [TinyLLaVA_Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory) variants.

## Prepare

1. Install [TinyLLaVA_Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory) and `llama.cpp`
2. Download and check model in `TinyLLaVA_Factory`. For example,
    ```bash
    hf download Zhang199/TinyLLaVA-Qwen2-0.5B-SigLIP
    ```
3. Convert `*.safetensor` to `*.gguf`. First modify the related path in `tools/mtmd/convert_TinyLLaVA_to_gguf.py#239`.
   Then run
    ```shell
    python convert_TinyLLaVA_to_gguf.py -m Zhang199/TinyLLaVA-Qwen2-0.5B-SigLIP -o <save-dir>
    ```

## Run

Build the `llama-mtmd-cli` binary.

After building, run: `./llama-mtmd-cli` to see the usage. For example:

```sh
cd <root>
<build_dir>/bin/llama-mtmd-cli -m <save_dir>/model-text-f16.gguf \
    --mmproj <save_dir>/model-vision-f16.gguf -v
```

Then input prompt to load an image:

```text
/image <image_path>
```

Now, you can chat with input image. For exmpale for Qwen2,

```text
How many people in the images?<|im_end|><|img_start|>
How many cars in the images?<|im_end|><|img_start|>
```
