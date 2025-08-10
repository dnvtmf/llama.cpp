import argparse
import sys
from torch import Tensor
from gguf import *
from tinyllava.model.modeling_tinyllava import TinyLlavaForConditionalGeneration

sys.path.append(Path(__file__).parent.parent.parent.as_posix())
from convert_hf_to_gguf import Qwen2Model

default_image_mean = [0.48145466, 0.4578275, 0.40821073]
default_image_std = [0.26862954, 0.26130258, 0.27577711]
TEXT = "clip.text"
VISION = "clip.vision"


def modify_tensors_qwen2(self: Qwen2Model, data_torch: Tensor, name: str, bid: int | None) -> Iterable[
    tuple[str, Tensor]]:
    if self.hf_arch == "Qwen2Model":
        name = f"model.{name}"  # map to Qwen2ForCausalLM tensors
    if "language_model." in name:
        name = name.replace("language_model.", "")  # for InternVL
    if name.startswith("vision_tower") or name.startswith("connector"):
        # skip vision and audio tensors
        return []
    yield from super(self.__class__, self).modify_tensors(data_torch, name, bid)


Qwen2Model.modify_tensors = modify_tensors_qwen2


def k(raw_key: str, arch: str) -> str:
    return raw_key.format(arch=arch)


def get_tensor_name(name: str) -> str:
    if name.startswith("connector"):
        name = name.replace("connector._connector", "mm")
        return name
    return name.replace(
        "text_model", "t").replace(
        "vision_tower._vision_tower.vision_model", "v").replace(
        "encoder.layers", "blk").replace(
        "embeddings.", "").replace(
        "_proj", "").replace(
        "self_attn.", "attn_").replace(
        "layer_norm", "ln").replace(
        "layernorm", "ln").replace(
        "mlp.fc1", "ffn_down").replace(
        "mlp.fc2", "ffn_up").replace(
        "embedding", "embd").replace(
        "final", "post").replace(
        "layrnorm", "ln")


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-name", help="Path to model directory cloned from HF Hub", required=True)
    parser.add_argument("-o", "--output-dir", help="Directory to save GGUF files. Default is the currect directory",
                        default=None)
    parser.add_argument("--use-f32", action="store_true", default=False, help="Use f32 instead of f16")
    parser.add_argument('--bigendian', action="store_true", default=False,
                        help="Model is executed on big-endian machine")
    # Example --image_mean 0.48145466 0.4578275 0.40821073 --image_std 0.26862954 0.26130258 0.27577711
    # Example --image_mean 0.5 0.5 0.5 --image_std 0.5 0.5 0.5
    parser.add_argument('--image-mean', type=float, nargs='+',
                        help='Mean of the images for normalization (overrides processor) ', default=None)
    parser.add_argument('--image-std', type=float, nargs='+',
                        help='Standard deviation of the images for normalization (overrides processor)', default=None)
    parser.add_argument('--verbose', action='store_true', default=False)
    # with proper
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    return args


def load_model(args):
    model = TinyLlavaForConditionalGeneration.from_pretrained(args.model_name, low_cpu_mem_usage=True)
    print(model)
    image_processor = model.vision_tower._image_processor
    context_len = getattr(model.config, 'max_sequence_length', 2048)
    tokenizer = model.tokenizer
    print(image_processor)
    print(context_len)
    print(tokenizer)
    return model, tokenizer, image_processor, context_len


def write_llm_info(fout, t_hparams, tokens, text_projection_dim=0):
    # text_model hparams
    fout.add_uint32(k(KEY_CONTEXT_LENGTH, TEXT), t_hparams["max_position_embeddings"])
    fout.add_uint32(k(KEY_EMBEDDING_LENGTH, TEXT), t_hparams["hidden_size"])
    fout.add_uint32(k(KEY_FEED_FORWARD_LENGTH, TEXT), t_hparams["intermediate_size"])
    # fout.add_uint32(f"{TEXT}.projection_dim", text_projection_dim)
    fout.add_uint32(k(KEY_ATTENTION_HEAD_COUNT, TEXT), t_hparams["num_attention_heads"])
    fout.add_float32(k(KEY_ATTENTION_LAYERNORM_EPS, TEXT), t_hparams.get("layer_norm_eps", 1e-6))
    fout.add_uint32(k(KEY_BLOCK_COUNT, TEXT), t_hparams["num_hidden_layers"])
    fout.add_token_list(tokens)
    return


def write_vision_tower_info(args, fout, config, visual_projection_dim=0):
    v_hparams = config.vision_config.to_dict()
    # set vision_model hparams
    fout.add_uint32(f"{VISION}.image_size", v_hparams["image_size"])
    fout.add_uint32(f"{VISION}.patch_size", v_hparams["patch_size"])
    fout.add_uint32(k(KEY_EMBEDDING_LENGTH, VISION), v_hparams["hidden_size"])
    fout.add_uint32(k(KEY_FEED_FORWARD_LENGTH, VISION), v_hparams["intermediate_size"])
    fout.add_uint32(f"{VISION}.projection_dim", visual_projection_dim)
    fout.add_uint32(k(KEY_ATTENTION_HEAD_COUNT, VISION), v_hparams["num_attention_heads"])
    fout.add_float32(k(KEY_ATTENTION_LAYERNORM_EPS, VISION), v_hparams.get("layer_norm_eps", 1e-6))
    # if feature_layers:
    #     block_count = max(feature_layers)
    # else:
    block_count = v_hparams["num_hidden_layers"] - 1  # if has_llava_projector else v_hparams["num_hidden_layers"]
    fout.add_uint32(k(KEY_BLOCK_COUNT, VISION), block_count)
    #     /**
    #      "image_grid_pinpoints": [
    #         [
    #         336,
    #         672
    #         ],
    #         [
    #         672,
    #         336
    #         ],
    #         [
    #         672,
    #         672
    #         ],
    #         [
    #         1008,
    #         336
    #         ],
    #         [
    #         336,
    #         1008
    #         ]
    #     ],
    #     Flattened:
    #     [
    #         336, 672,
    #         672, 336,
    #         672, 672,
    #         1008, 336,
    #         336, 1008
    #     ]
    #  *
    #  */
    # if "image_grid_pinpoints" in v_hparams:
    #     # flatten it
    #     image_grid_pinpoints = []
    #     for pinpoint in v_hparams["image_grid_pinpoints"]:
    #         for p in pinpoint:
    #             image_grid_pinpoints.append(p)
    #     fout.add_array(f"{VISION}.image_grid_pinpoints", image_grid_pinpoints)
    # if "image_crop_resolution" in v_hparams:
    #     fout.add_uint32(f"{VISION}.image_crop_resolution", v_hparams["image_crop_resolution"])
    # if "image_aspect_ratio" in v_hparams:
    #     fout.add_string(f"{VISION}.image_aspect_ratio", v_hparams["image_aspect_ratio"])
    # if "image_split_resolution" in v_hparams:
    #     fout.add_uint32(f"{VISION}.image_split_resolution", v_hparams["image_split_resolution"])
    # if "mm_patch_merge_type" in v_hparams:
    #     fout.add_string(f"{VISION}.mm_patch_merge_type", v_hparams["mm_patch_merge_type"])
    # if "mm_projector_type" in v_hparams:
    #     fout.add_string(f"{VISION}.mm_projector_type", v_hparams["mm_projector_type"])
    # if feature_layers:
    #     fout.add_array(f"{VISION}.feature_layer", feature_layers)

    # if processor is not None:
    #     image_mean = processor.image_processor.image_mean if args.image_mean is None or args.image_mean == default_image_mean else args.image_mean  # pyright: ignore[reportAttributeAccessIssue]
    #     image_std = processor.image_processor.image_std if args.image_std is None or args.image_std == default_image_std else args.image_std  # pyright: ignore[reportAttributeAccessIssue]
    # else:
    image_mean = args.image_mean if args.image_mean is not None else default_image_mean
    image_std = args.image_std if args.image_std is not None else default_image_std
    fout.add_array(f"{VISION}.image_mean", image_mean)
    fout.add_array(f"{VISION}.image_std", image_std)

    use_gelu = v_hparams["hidden_act"] == "gelu"
    fout.add_bool("clip.use_gelu", use_gelu)


def write_tensors(fout, model, ftype, ftype_str):
    state_dict = model.state_dict()
    for name, data in state_dict.items():

        name = get_tensor_name(name)
        data = data.squeeze().numpy()

        n_dims = len(data.shape)

        # ftype == 0 -> float32, ftype == 1 -> float16
        ftype_cur = 0
        convert_s = ''
        if n_dims == 4:
            print(f"tensor {name} is always saved in f16")
            data = data.astype(np.float16)
            ftype_cur = 1
        elif ftype == 1:
            if name[-7:] == ".weight" and n_dims == 2:
                convert_s = ("  Converting to float16")
                data = data.astype(np.float16)
                ftype_cur = 1
            else:
                convert_s = ("  Converting to float32")
                data = data.astype(np.float32)
                ftype_cur = 0
        else:
            if data.dtype != np.float32:
                convert_s = ("  Converting to float32")
                data = data.astype(np.float32)
                ftype_cur = 0

        print(f"{name} - {ftype_str[ftype_cur]} - shape = {data.shape}{convert_s}")
        fout.add_tensor(name, data)


def main():
    args = options()
    model, tokenizer, image_processor, context_len = load_model(args)
    config = model.config
    print(config)
    ftype_map: dict[str, gguf.LlamaFileType] = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "tq1_0": gguf.LlamaFileType.MOSTLY_TQ1_0,
        "tq2_0": gguf.LlamaFileType.MOSTLY_TQ2_0,
        "auto": gguf.LlamaFileType.GUESSED,
    }
    ftype_str = ["f32", "f16"]
    ftype = 1
    if args.use_f32:
        ftype = 0
    model_dir = Path(
        '~/.cache/huggingface/hub/models--Zhang199--TinyLLaVA-Qwen2-0.5B-SigLIP/snapshots/6aef66ed2e0125f57a5ec562fe3c0bf1204d8fa3').expanduser()

    output_dir = Path(args.output_dir) if args.output_dir is not None else Path('.')
    os.makedirs(output_dir, exist_ok=True)
    output_prefix = os.path.basename(output_dir).replace("ggml_", "")

    ## convert language model to gguf
    fname_out = output_dir / f"model-text-{ftype_str[ftype]}.gguf"
    instance = Qwen2Model(
        model_dir, ftype_map[ftype_str[ftype]], fname_out
    )
    # instance.hparams.setdefault('layer_norm_rms_eps', 1e-6)
    # t_hparams = config.text_config.to_dict()
    # if 'rms_norm_eps' not in t_hparams:
    instance.gguf_writer.add_float32(k(KEY_ATTENTION_LAYERNORM_RMS_EPS, 'qwen2'), 1e-6)
    logger.info("Exporting model...")
    instance.write()
    print(instance.hparams)
    logger.info(f"Model successfully exported to {fname_out}")
    # return
    ##
    fname_out = output_dir / f"model-vision-{ftype_str[ftype]}.gguf"
    vocab_path = model_dir / 'vocab.json'
    # tokens = [key for key in tokenizer.get_vocab()]
    with vocab_path.open('r') as f:
        vocab = json.load(f)
    tokens = [key for key in vocab]  # tokenizer.get_vocab()]
    print('tokens:', len(tokens), config.vocab_size, tokens[:10], tokens[-10:])
    print(tokenizer.all_special_ids)

    fout = GGUFWriter(path=fname_out, arch="clip",
                      endianess=GGUFEndian.LITTLE if not args.bigendian else GGUFEndian.BIG)
    fout.add_bool("clip.has_text_encoder", True)
    fout.add_bool("clip.has_vision_encoder", True)
    fout.add_bool("clip.has_llava_projector", False)
    fout.add_file_type(ftype)
    model_name = os.path.basename(config.name_or_path)
    print(f"{model_name=}")
    # if "_name_or_path" in config else os.path.basename(args.model_name)
    fout.add_name(model_name)
    fout.add_string("clip.projector_type", 'mlp')

    write_llm_info(fout, config.text_config.to_dict(), tokens)
    write_vision_tower_info(args, fout, config)
    write_tensors(fout, model, ftype, ftype_str)

    fout.write_header_to_file()
    fout.write_kv_data_to_file()
    fout.write_tensors_to_file()
    fout.close()
    print("Done. Output file: ", fname_out)
    return


if __name__ == '__main__':
    main()
