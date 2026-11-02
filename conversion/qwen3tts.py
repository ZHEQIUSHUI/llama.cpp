from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, MmprojModel, TextModel, gguf

# Tricks being used to support this model via existing llama.cpp code paths:
# - Text projection MLP is folded into the embedding table
# - codec_embedding is concat to the text embedding table, vocab is extended
#   example: codec_bos_id(2149) --> "<|codec_bos|>"
#            codec_eos_token_id(2150) --> "<|codec_eos_token|>"
#            codec_language_id.chinese(2055) --> "<|codec_language_chinese|>"
#            other rows --> "<|codec_0|>", "<|codec_1|>", ..., "<|codec_1023|>"
# - output tensor codec_head is smaller than vocab, so logits will be padded at inference time
# - suppress_tokens is used to limit the backbone to only sample either semantic or EOS (stop) token

# pipeline stage mapping:
#   speaker reference encoder --> mapped to normal mtmd audio encoder
#   backbone --> mapped to normal libllama text model (autoregressive)
#   code_predictor --> MTMD_GEN_PROCESS_TYPE_GEN_CODE
#   code2wav --> MTMD_GEN_PROCESS_TYPE_GEN_WAV

# torch activation functions used by Qwen3TTSTalkerResizeMLP (config's hidden_act)
_ACT2FN = {
    "silu": F.silu,
    "gelu": F.gelu,
    "relu": F.relu,
}


@ModelBase.register("Qwen3TTSForConditionalGeneration")
@ModelBase.example("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
class Qwen3TTSTalkerModel(TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN3TTS

    _TEXT_PROJ_KEYS = (
        "model.text_embedding.weight",
        "text_projection.linear_fc1.weight",
        "text_projection.linear_fc1.bias",
        "text_projection.linear_fc2.weight",
        "text_projection.linear_fc2.bias",
    )

    _text_proj_buffer: dict[str, Tensor]
    _folded_text_embed: Tensor | None
    _codec_embed: Tensor | None

    def __init__(self, dir_model: Path, *args, **kwargs):
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            hparams = ModelBase.load_hparams(dir_model, is_mistral_format=False)
        raw_talker_config = dict(hparams["talker_config"])
        self._talker_config = raw_talker_config
        self.n_codec_vocab = raw_talker_config["vocab_size"]
        talker_config = dict(raw_talker_config)
        talker_config["vocab_size"] = talker_config["text_vocab_size"]
        hparams["text_config"] = talker_config
        super().__init__(dir_model, *args, hparams=hparams, **kwargs)
        self._text_proj_buffer = {}
        self._folded_text_embed = None
        self._codec_embed = None

    def _codec_token_names(self) -> list[str]:
        # start every row with a generic name, then override the ones with a
        # known meaning (bos/eos/language/etc, derived from the *_id fields
        # of talker_config) with a more descriptive one
        names = [f"<|codec_{i}|>" for i in range(self.n_codec_vocab)]
        for key, val in self._talker_config.items():
            if not key.endswith("_id"):
                continue
            prefix = key[:-len("_id")]
            if isinstance(val, int):
                names[val] = f"<|{prefix}|>"
            elif isinstance(val, dict):
                for subkey, subval in val.items():
                    names[subval] = f"<|{prefix}_{subkey}|>"
        return names

    def set_vocab(self):
        codec_tokens = self._codec_token_names()
        codec_toktypes = [gguf.TokenType.CONTROL] * len(codec_tokens)

        try:
            tokens, scores, toktypes = self._create_vocab_sentencepiece()
            self.gguf_writer.add_tokenizer_model("llama")
            self.gguf_writer.add_tokenizer_pre("default")
            tokens += [t.encode("utf-8") for t in codec_tokens]
            scores += [0.0] * len(codec_tokens)
            toktypes += codec_toktypes
            self.gguf_writer.add_token_list(tokens)
            self.gguf_writer.add_token_scores(scores)
            self.gguf_writer.add_token_types(toktypes)
            special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
            special_vocab.add_to_gguf(self.gguf_writer)
            return
        except FileNotFoundError:
            pass

        tokens, toktypes, tokpre = self.get_vocab_base()
        tokens += codec_tokens
        toktypes += codec_toktypes
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

        # make sure that the model has no chat template, so chat will be disabled
        self.gguf_writer.add_chat_template(None)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        # note: final vocab layout is [text_vocab | codec_vocab], with text_vocab is actually padded with -inf in cgraph
        # for codec_vocab, only first 2048 rows can be sampled for semantic code
        # plus codec_eos_token_id that used for signaling end of generation
        # ref: https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/models/modeling_qwen3_tts.py#L2059-L2063

        vocab_size = self.hparams["vocab_size"] + self.n_codec_vocab
        codec_eos_token_id = self.hparams["vocab_size"] + self._talker_config["codec_eos_token_id"]
        self.gguf_writer.add_suppress_tokens([
            i for i in range(vocab_size - 1024, vocab_size)
            if i != codec_eos_token_id
        ])
        self.gguf_writer.add_eos_token_id(codec_eos_token_id)
        self.gguf_writer.add_add_eos_token(False)

        # The three 12 Hz variants share the talker/code-predictor/codec graphs and
        # differ only in the prompt. Keep the prompt-selection data next to the
        # talker weights so llama-tts can select the right mode from the GGUF alone.
        tts_model_type = self.hparams.get("tts_model_type", "base")
        self.gguf_writer.add_string("qwen3tts.tts_model_type", tts_model_type)
        self.gguf_writer.add_string("qwen3tts.tts_model_size", self.hparams.get("tts_model_size", ""))

        talker_config = self.hparams.get("talker_config", {})
        spk_id = talker_config.get("spk_id") or {}
        spk_is_dialect = talker_config.get("spk_is_dialect") or {}
        for speaker, token_id in spk_id.items():
            speaker = speaker.lower()
            self.gguf_writer.add_uint32(f"qwen3tts.spk_id.{speaker}", token_id)
            dialect = spk_is_dialect.get(speaker, False)
            self.gguf_writer.add_string(f"qwen3tts.spk_dialect.{speaker}", dialect or "")

        for language, token_id in (talker_config.get("codec_language_id") or {}).items():
            self.gguf_writer.add_uint32(f"qwen3tts.codec_language_id.{language.lower()}", token_id)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if not name.startswith("talker.") or name.startswith("talker.code_predictor."):
            return None

        name = name[len("talker."):]
        return super().filter_tensors((name, gen))

    def _maybe_emit_token_embd(self) -> Iterable[tuple[str, Tensor]]:
        if self._folded_text_embed is None or self._codec_embed is None:
            return
        combined = torch.cat([self._folded_text_embed, self._codec_embed], dim=0)
        yield (self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD), combined)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # codec_embedding rows are appended after the text vocab, extending the embedding table
        if name == "model.codec_embedding.weight":
            self._codec_embed = data_torch
            yield from self._maybe_emit_token_embd()
            return

        # codec_head is the output head for the (smaller) codec vocab; logits get padded to
        # the extended vocab size at inference time
        if name == "codec_head.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT), data_torch)
            return

        if name in self._TEXT_PROJ_KEYS:
            self._text_proj_buffer[name] = data_torch
            if len(self._text_proj_buffer) < len(self._TEXT_PROJ_KEYS):
                return

            # fold MLP into the embedding table at conversion time, MLP won't be used at inference time anyway
            act_fn = _ACT2FN[self.hparams["hidden_act"]]
            embed = self._text_proj_buffer["model.text_embedding.weight"]
            hidden = act_fn(F.linear(embed,
                                     self._text_proj_buffer["text_projection.linear_fc1.weight"],
                                     self._text_proj_buffer["text_projection.linear_fc1.bias"]))
            folded = F.linear(hidden,
                              self._text_proj_buffer["text_projection.linear_fc2.weight"],
                              self._text_proj_buffer["text_projection.linear_fc2.bias"])
            self._folded_text_embed = folded
            yield from self._maybe_emit_token_embd()
            return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Qwen3TTSForConditionalGeneration")
@ModelBase.example("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
class Qwen3TTSSpeakerEncoderModel(MmprojModel):
    has_vision_encoder = False
    has_audio_encoder = True

    # talker.code_predictor.model.layers.{bid}.<key> -> A_GEN_CODE_*
    # bypass tensor_mapping.py for now to make it simple
    _CODE_LAYER_TENSOR_MAP = {
        "input_layernorm": gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_NORM,
        "self_attn.q_proj": gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_Q,
        "self_attn.q_norm": gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_Q_NORM,
        "self_attn.k_proj": gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_K,
        "self_attn.k_norm": gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_K_NORM,
        "self_attn.v_proj": gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_V,
        "self_attn.o_proj": gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_OUT,
        "post_attention_layernorm": gguf.MODEL_TENSOR.A_GEN_CODE_FFN_NORM,
        "mlp.gate_proj": gguf.MODEL_TENSOR.A_GEN_CODE_FFN_GATE,
        "mlp.up_proj": gguf.MODEL_TENSOR.A_GEN_CODE_FFN_UP,
        "mlp.down_proj": gguf.MODEL_TENSOR.A_GEN_CODE_FFN_DOWN,
    }

    # note: codebook pages will be stacked to 3D
    _CODE_GEN_N_CODEBOOKS = 15
    _code_embed_buffer: dict[int, Tensor] = {}
    _code_head_buffer: dict[int, Tensor] = {}
    _wav_config_cache: dict[str, Any] | None = None

    def __init__(self, dir_model: Path, *args, **kwargs):
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            hparams = ModelBase.load_hparams(dir_model, is_mistral_format=False)
        hparams["text_config"] = {"hidden_size": hparams["talker_config"]["hidden_size"]}
        # ECAPA-TDNN has a fixed 4-stage backbone, but MmprojModel.__init__ needs a n_block_keys.
        # The CustomVoice variant ships no speaker encoder, so its config lacks this key entirely.
        self._speaker_encoder_present = "speaker_encoder_config" in hparams
        self.has_audio_encoder = self._speaker_encoder_present
        if "speaker_encoder_config" in hparams:
            hparams["speaker_encoder_config"]["n_layers"] = 4
        super().__init__(dir_model, *args, hparams=hparams, **kwargs)
        self._load_encoder_tensors()
        self._wav_config_cache = None

    def _load_encoder_tensors(self) -> None:
        # CustomVoice / VoiceDesign have no ICL voice-clone encoder. Their mmproj
        # carries only the code predictor and code2wav decoder.
        if self.global_config.get("tts_model_type", "base") != "base":
            return
        # The 12 Hz tokenizer lives in a separate safetensors file under speech_tokenizer/.
        # Keep it lazy: conversion already walks a large talker checkpoint, and only the
        # encoder (not the code2wav decoder) is needed in the mmproj.
        path = self.dir_model / "speech_tokenizer" / "model.safetensors"
        if not path.is_file():
            return
        part = gguf.utility.SafetensorsLocal(path)
        for name, data in part.tensors.items():
            if not name.startswith("encoder.encoder") and not name.startswith("encoder.encoder_transformer"):
                if not name.startswith("encoder.downsample") and not name.startswith("encoder.quantizer"):
                    continue
            if self.lazy:
                gen = lambda data=data: LazyTorchTensor.from_local_tensor(data)  # noqa: E731
            else:
                dtype = LazyTorchTensor._dtype_str_map[data.dtype]
                gen = lambda data=data, dtype=dtype: torch.from_numpy(data.mmap_bytes()).view(dtype).reshape(data.shape)  # noqa: E731
            self.model_tensors[name] = gen

    def get_audio_config(self) -> dict[str, Any] | None:
        # MmprojModel requires a non-empty audio config even when this variant has
        # no speaker encoder. A single dummy layer satisfies its tensor-name map;
        # the code predictor and code2wav tensors are emitted explicitly below.
        return self.global_config.get("speaker_encoder_config") or {"n_layers": 1}

    def set_gguf_parameters(self):
        self.gguf_writer.add_file_type(self.ftype)
        if self._speaker_encoder_present:
            self.gguf_writer.add_clip_has_audio_encoder(True)
            self.gguf_writer.add_clip_audio_projector_type(gguf.VisionProjectorType.QWEN3TTS_SPKENC)

            # handle speaker encoder config
            self.gguf_writer.add_audio_projection_dim(self.n_embd_text)
            # mel_spectrogram() front-end: sr=24000, n_fft=1024, hop=256, n_mels=128, fmin=0, fmax=12000 (=sr/2, the clip.cpp default)
            self.gguf_writer.add_audio_num_mel_bins(128)
            # 3 SE-Res2Net stages; the stem conv, mfa, asp and fc are not counted here
            self.gguf_writer.add_audio_block_count(3)
            # ECAPA-TDNN has no attention/FFN, these are dummy to allow clip.cpp to load it
            self.gguf_writer.add_audio_embedding_length(1536)
            self.gguf_writer.add_audio_head_count(1)
            self.gguf_writer.add_audio_feed_forward_length(1536)
            self.gguf_writer.add_audio_attention_layernorm_eps(1e-5)

        # handle code predictor config
        self.gguf_writer.add_clip_has_gen_audio_encoder(True)
        self.gguf_writer.add_clip_gen_audio_projector_type(gguf.VisionProjectorType.QWEN3TTS_GEN)
        code_predictor_config = self.global_config["talker_config"]["code_predictor_config"]
        self.gguf_writer.add_gen_audio_projection_dim(self.n_embd_text)
        self.gguf_writer.add_gen_audio_embedding_length(code_predictor_config["hidden_size"])
        self.gguf_writer.add_gen_audio_feed_forward_length(code_predictor_config["intermediate_size"])
        self.gguf_writer.add_gen_audio_block_count(code_predictor_config["num_hidden_layers"])
        self.gguf_writer.add_gen_audio_head_count(code_predictor_config["num_attention_heads"])
        self.gguf_writer.add_gen_audio_head_count_kv(code_predictor_config["num_key_value_heads"])
        self.gguf_writer.add_gen_audio_attention_layernorm_eps(code_predictor_config["rms_norm_eps"])
        # note: code2wav hparams are hardcoded on the mtmd/clip.cpp side for now, not written here

    def _wav_decoder_config(self) -> dict[str, Any] | None:
        # code2wav has its own config.json, inside the speech_tokenizer dir
        if self._wav_config_cache is None:
            path = self.dir_model / "speech_tokenizer" / "config.json"
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self._wav_config_cache = cfg["decoder_config"]
        return self._wav_config_cache

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # the speech tokenizer encoder is sensitive to weight rounding: residual RVQ makes
        # small hidden-state errors change later codebook indices. Keep its payload in F32;
        # all pre-existing mmproj tensors still follow the requested outtype (BF16).
        if new_name.startswith("a.enc."):
            return gguf.GGMLQuantizationType.F32
        # conv1d/conv1d_dw kernels must be F16, ggml_conv_1d(_dw) has no BF16 path
        if new_name.endswith(".weight") and (
            new_name in ("a.gen.wav.pre_conv.weight", "a.gen.wav.dac.entry.weight", "a.gen.wav.dac.post_conv.weight")
            or (".up.blk." in new_name and new_name.endswith(".dwconv.weight"))
            or (".dac.blk." in new_name and (new_name.endswith(".conv1.weight") or new_name.endswith(".conv2.weight")))
        ):
            return gguf.GGMLQuantizationType.F16
        # ConvTranspose1d kernels: only F16/F32 are implemented, no BF16
        if new_name.endswith(".conv.weight") and (".up.blk." in new_name or ".dac.blk." in new_name):
            return gguf.GGMLQuantizationType.F32
        # the code predictor FFN intermediate peaks around 1.5e5, above the F16 range, and mul_mat
        # casts its input to the weight type
        if new_name.startswith("a.gen.code.blk.") and new_name.endswith(".ffn_down.weight"):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if not (
            name.startswith("speaker_encoder.")
            or name.startswith("encoder.encoder")
            or name.startswith("encoder.downsample.")
            or name.startswith("encoder.quantizer.")
            or name.startswith("talker.code_predictor.")
            or name == "talker.model.codec_embedding.weight"
        ):
            return None

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # code2wav tensors are already named by generate_extra_tensors(), pass them through
        if name.startswith("a.gen.wav."):
            yield (name, data_torch)
            return

        # 12 Hz speech tokenizer encoder: SEANet + transformer + 16 RVQ codebooks.
        if name.startswith("encoder."):
            yield from self._modify_encoder_tensor(data_torch, name)
            return

        # codebook-0 embedding, fed back to the talker backbone (codebooks 1-15 live in code_predictor)
        if name == "talker.model.codec_embedding.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.A_GEN_CODE_OUT_EMBD), data_torch)
            return

        if name == "talker.code_predictor.model.norm.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.A_GEN_CODE_OUTPUT_NORM), data_torch)
            return

        if name.startswith("talker.code_predictor.small_to_mtp_projection."):
            suffix = "." + name.rsplit(".", 1)[1]
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.A_GEN_CODE_PROJ_IN, suffix=suffix), data_torch)
            return

        if name.startswith("talker.code_predictor.model.codec_embedding."):
            idx = int(name.split("codec_embedding.")[1].split(".")[0])
            self._code_embed_buffer[idx] = data_torch
            if len(self._code_embed_buffer) < self._CODE_GEN_N_CODEBOOKS:
                return
            stacked = torch.stack([self._code_embed_buffer.pop(i) for i in range(self._CODE_GEN_N_CODEBOOKS)], dim=0)
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.A_GEN_CODE_EMBD), stacked)
            return

        if name.startswith("talker.code_predictor.lm_head."):
            idx = int(name.split("lm_head.")[1].split(".")[0])
            self._code_head_buffer[idx] = data_torch
            if len(self._code_head_buffer) < self._CODE_GEN_N_CODEBOOKS:
                return
            stacked = torch.stack([self._code_head_buffer.pop(i) for i in range(self._CODE_GEN_N_CODEBOOKS)], dim=0)
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.A_GEN_CODE_HEAD), stacked)
            return

        if name.startswith("talker.code_predictor.model.layers."):
            rest = name.split("model.layers.")[1]        # "{bid}.<key>.weight"
            _, key_with_suffix = rest.split(".", 1)       # "<key>.weight"
            key = key_with_suffix.rsplit(".", 1)[0]        # "<key>"
            tensor = self._CODE_LAYER_TENSOR_MAP.get(key)
            if tensor is not None:
                yield (self.format_tensor_name(tensor, bid), data_torch)
                return

        if "res2net_block.blocks." in name:
            assert bid is not None  # the outer stage index, picked up from the tensor name automatically
            xid = int(name.split("res2net_block.blocks.")[1].split(".")[0])
            suffix = "." + name.rsplit(".", 1)[1]
            new_name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.A_ENC_CONV_RES2].format(bid=bid, xid=xid) + suffix
            yield (new_name, data_torch)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def _modify_encoder_tensor(self, data_torch: Tensor, name: str) -> Iterable[tuple[str, Tensor]]:
        if name == "encoder.downsample.conv.weight":
            yield ("a.enc.down.weight", data_torch)
            return

        if name.startswith("encoder.encoder.layers."):
            rest = name[len("encoder.encoder.layers."):]
            layer_s, key = rest.split(".", 1)
            layer = int(layer_s)
            if key.startswith("block.1.conv."):
                yield (f"a.enc.seanet.{layer}.res1.{key.rsplit('.', 1)[1]}", data_torch)
            elif key.startswith("block.3.conv."):
                yield (f"a.enc.seanet.{layer}.res2.{key.rsplit('.', 1)[1]}", data_torch)
            elif key.startswith("conv."):
                yield (f"a.enc.seanet.{layer}.conv.{key.rsplit('.', 1)[1]}", data_torch)
            return

        if name.startswith("encoder.encoder_transformer.layers."):
            rest = name[len("encoder.encoder_transformer.layers."):]
            layer_s, key = rest.split(".", 1)
            layer = int(layer_s)
            key_map = {
                "input_layernorm.weight": "attn_norm.weight",
                "input_layernorm.bias": "attn_norm.bias",
                "post_attention_layernorm.weight": "ffn_norm.weight",
                "post_attention_layernorm.bias": "ffn_norm.bias",
                "self_attn.q_proj.weight": "attn_q.weight",
                "self_attn.k_proj.weight": "attn_k.weight",
                "self_attn.v_proj.weight": "attn_v.weight",
                "self_attn.o_proj.weight": "attn_o.weight",
                "self_attn_layer_scale.scale": "attn_scale",
                "mlp.fc1.weight": "ffn_up.weight",
                "mlp.fc2.weight": "ffn_down.weight",
                "mlp_layer_scale.scale": "ffn_scale",
            }
            new_key = key_map.get(key)
            if new_key is not None:
                yield (f"a.enc.tfm.blk.{layer}.{new_key}", data_torch)
            return

        if name == "encoder.quantizer.semantic_residual_vector_quantizer.input_proj.weight":
            yield ("a.enc.quant.sem.in_proj.weight", data_torch)
            return

        if name == "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum":
            usage = self.model_tensors["encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.cluster_usage"]()
            yield ("a.enc.quant.sem.codebook.weight", data_torch / usage.clamp(min=1e-5)[:, None])
            return

        if name == "encoder.quantizer.acoustic_residual_vector_quantizer.input_proj.weight":
            yield ("a.enc.quant.aco.in_proj.weight", data_torch)
            return

        if name.startswith("encoder.quantizer.acoustic_residual_vector_quantizer.layers."):
            rest = name[len("encoder.quantizer.acoustic_residual_vector_quantizer.layers."):]
            layer_s, key = rest.split(".", 1)
            layer = int(layer_s)
            if key == "codebook.embed_sum" and layer < 15:
                usage = self.model_tensors[
                    f"encoder.quantizer.acoustic_residual_vector_quantizer.layers.{layer}.codebook.cluster_usage"]()
                yield (f"a.enc.quant.aco.codebook.{layer:02d}.weight", data_torch / usage.clamp(min=1e-5)[:, None])
            return

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        yield from self._generate_code2wav_tensors()

    def _generate_code2wav_tensors(self) -> Iterable[tuple[str, Tensor]]:
        # code2wav weights live in speech_tokenizer/model.safetensors, not the main safetensors
        from safetensors.torch import load_file

        wav_config = self._wav_decoder_config()
        state_dict = load_file(self.dir_model / "speech_tokenizer" / "model.safetensors")

        def get(name: str) -> Tensor:
            return state_dict[name]

        def snake_fold(alpha: Tensor, beta: Tensor) -> tuple[Tensor, Tensor]:
            # fold SnakeBeta's exp()/reciprocal here, so the graph is only mul/sin/sqr/mul/add
            return torch.exp(alpha), 1.0 / (torch.exp(beta) + 1e-9)

        def rvq_codebook(prefix: str, n_layers: int) -> Tensor:
            # checkpoint has EMA accumulators, so codebook[i] = embedding_sum[i] / cluster_usage[i]
            books = []
            for i in range(n_layers):
                embedding_sum = get(f"{prefix}.vq.layers.{i}._codebook.embedding_sum")
                cluster_usage = get(f"{prefix}.vq.layers.{i}._codebook.cluster_usage")
                books.append(embedding_sum / cluster_usage.clamp_min(1e-5).unsqueeze(-1))
            return torch.stack(books, dim=0) if n_layers > 1 else books[0]

        T = gguf.MODEL_TENSOR

        # --- quantizer: RVQ codebook decode ---
        yield (self.format_tensor_name(T.A_GEN_WAV_QUANT_FIRST_IN), get("decoder.quantizer.rvq_first.input_proj.weight").squeeze(-1))
        yield (self.format_tensor_name(T.A_GEN_WAV_QUANT_FIRST_OUT), get("decoder.quantizer.rvq_first.output_proj.weight").squeeze(-1))
        yield (self.format_tensor_name(T.A_GEN_WAV_QUANT_FIRST_CB), rvq_codebook("decoder.quantizer.rvq_first", 1))
        yield (self.format_tensor_name(T.A_GEN_WAV_QUANT_REST_IN), get("decoder.quantizer.rvq_rest.input_proj.weight").squeeze(-1))
        yield (self.format_tensor_name(T.A_GEN_WAV_QUANT_REST_OUT), get("decoder.quantizer.rvq_rest.output_proj.weight").squeeze(-1))
        yield (self.format_tensor_name(T.A_GEN_WAV_QUANT_REST_CB), rvq_codebook("decoder.quantizer.rvq_rest", self._CODE_GEN_N_CODEBOOKS))

        # --- pre_conv ---
        yield (self.format_tensor_name(T.A_GEN_WAV_PRE_CONV, suffix=".weight"), get("decoder.pre_conv.conv.weight"))
        yield (self.format_tensor_name(T.A_GEN_WAV_PRE_CONV, suffix=".bias"), get("decoder.pre_conv.conv.bias"))

        # --- pre_transformer ---
        yield (self.format_tensor_name(T.A_GEN_WAV_TFM_IN_PROJ, suffix=".weight"), get("decoder.pre_transformer.input_proj.weight"))
        yield (self.format_tensor_name(T.A_GEN_WAV_TFM_IN_PROJ, suffix=".bias"), get("decoder.pre_transformer.input_proj.bias"))
        yield (self.format_tensor_name(T.A_GEN_WAV_TFM_OUT_PROJ, suffix=".weight"), get("decoder.pre_transformer.output_proj.weight"))
        yield (self.format_tensor_name(T.A_GEN_WAV_TFM_OUT_PROJ, suffix=".bias"), get("decoder.pre_transformer.output_proj.bias"))
        yield (self.format_tensor_name(T.A_GEN_WAV_TFM_OUTPUT_NORM), get("decoder.pre_transformer.norm.weight"))

        tfm_layer_map = {
            "input_layernorm.weight":         T.A_GEN_WAV_TFM_ATTN_NORM,
            "self_attn.q_proj.weight":        T.A_GEN_WAV_TFM_ATTN_Q,
            "self_attn.k_proj.weight":        T.A_GEN_WAV_TFM_ATTN_K,
            "self_attn.v_proj.weight":        T.A_GEN_WAV_TFM_ATTN_V,
            "self_attn.o_proj.weight":        T.A_GEN_WAV_TFM_ATTN_OUT,
            "self_attn_layer_scale.scale":    T.A_GEN_WAV_TFM_ATTN_SCALE,
            "post_attention_layernorm.weight": T.A_GEN_WAV_TFM_FFN_NORM,
            "mlp.gate_proj.weight":           T.A_GEN_WAV_TFM_FFN_GATE,
            "mlp.up_proj.weight":             T.A_GEN_WAV_TFM_FFN_UP,
            "mlp.down_proj.weight":           T.A_GEN_WAV_TFM_FFN_DOWN,
            "mlp_layer_scale.scale":          T.A_GEN_WAV_TFM_FFN_SCALE,
        }
        assert wav_config is not None
        for bid in range(wav_config["num_hidden_layers"]):
            for key, tensor_id in tfm_layer_map.items():
                yield (self.format_tensor_name(tensor_id, bid), get(f"decoder.pre_transformer.layers.{bid}.{key}"))

        # --- upsample: 2x (causal ConvTranspose1d + ConvNeXt block) ---
        up_map = {
            "0.conv.weight":     (T.A_GEN_WAV_UP_CONV, ".weight"),
            "0.conv.bias":       (T.A_GEN_WAV_UP_CONV, ".bias"),
            "1.dwconv.conv.weight": (T.A_GEN_WAV_UP_DWCONV, ".weight"),
            "1.dwconv.conv.bias":   (T.A_GEN_WAV_UP_DWCONV, ".bias"),
            "1.norm.weight":     (T.A_GEN_WAV_UP_NORM, ".weight"),
            "1.norm.bias":       (T.A_GEN_WAV_UP_NORM, ".bias"),
            "1.pwconv1.weight":  (T.A_GEN_WAV_UP_PW1, ".weight"),
            "1.pwconv1.bias":    (T.A_GEN_WAV_UP_PW1, ".bias"),
            "1.pwconv2.weight":  (T.A_GEN_WAV_UP_PW2, ".weight"),
            "1.pwconv2.bias":    (T.A_GEN_WAV_UP_PW2, ".bias"),
            "1.gamma":           (T.A_GEN_WAV_UP_GAMMA, ""),
        }
        for bid in range(len(wav_config["upsampling_ratios"])):
            for key, (tensor_id, suffix) in up_map.items():
                yield (self.format_tensor_name(tensor_id, bid, suffix=suffix), get(f"decoder.upsample.{bid}.{key}"))

        # --- DAC decoder ---
        yield (self.format_tensor_name(T.A_GEN_WAV_DAC_ENTRY, suffix=".weight"), get("decoder.decoder.0.conv.weight"))
        yield (self.format_tensor_name(T.A_GEN_WAV_DAC_ENTRY, suffix=".bias"), get("decoder.decoder.0.conv.bias"))

        n_dac_blocks = len(wav_config["upsample_rates"])
        for bid in range(n_dac_blocks):
            py = bid + 1  # decoder.decoder.0 is the entry conv, blocks start at 1

            a, b = snake_fold(get(f"decoder.decoder.{py}.block.0.alpha"), get(f"decoder.decoder.{py}.block.0.beta"))
            yield (self.format_tensor_name(T.A_GEN_WAV_DAC_UP_SNAKE, bid, suffix=".alpha"), a)
            yield (self.format_tensor_name(T.A_GEN_WAV_DAC_UP_SNAKE, bid, suffix=".beta"), b)
            yield (self.format_tensor_name(T.A_GEN_WAV_DAC_UP_CONV, bid, suffix=".weight"), get(f"decoder.decoder.{py}.block.1.conv.weight"))
            yield (self.format_tensor_name(T.A_GEN_WAV_DAC_UP_CONV, bid, suffix=".bias"), get(f"decoder.decoder.{py}.block.1.conv.bias"))

            for xid in range(3):
                ridx = xid + 2  # block.2/3/4 are the 3 residual units

                a1, b1 = snake_fold(get(f"decoder.decoder.{py}.block.{ridx}.act1.alpha"), get(f"decoder.decoder.{py}.block.{ridx}.act1.beta"))
                name1 = gguf.TENSOR_NAMES[T.A_GEN_WAV_DAC_RES_ACT1].format(bid=bid, xid=xid)
                yield (name1 + ".alpha", a1)
                yield (name1 + ".beta", b1)

                name_conv1 = gguf.TENSOR_NAMES[T.A_GEN_WAV_DAC_RES_CONV1].format(bid=bid, xid=xid)
                yield (name_conv1 + ".weight", get(f"decoder.decoder.{py}.block.{ridx}.conv1.conv.weight"))
                yield (name_conv1 + ".bias", get(f"decoder.decoder.{py}.block.{ridx}.conv1.conv.bias"))

                a2, b2 = snake_fold(get(f"decoder.decoder.{py}.block.{ridx}.act2.alpha"), get(f"decoder.decoder.{py}.block.{ridx}.act2.beta"))
                name2 = gguf.TENSOR_NAMES[T.A_GEN_WAV_DAC_RES_ACT2].format(bid=bid, xid=xid)
                yield (name2 + ".alpha", a2)
                yield (name2 + ".beta", b2)

                name_conv2 = gguf.TENSOR_NAMES[T.A_GEN_WAV_DAC_RES_CONV2].format(bid=bid, xid=xid)
                yield (name_conv2 + ".weight", get(f"decoder.decoder.{py}.block.{ridx}.conv2.conv.weight"))
                yield (name_conv2 + ".bias", get(f"decoder.decoder.{py}.block.{ridx}.conv2.conv.bias"))

        a5, b5 = snake_fold(get("decoder.decoder.5.alpha"), get("decoder.decoder.5.beta"))
        yield (self.format_tensor_name(T.A_GEN_WAV_DAC_POST_SNAKE, suffix=".alpha"), a5)
        yield (self.format_tensor_name(T.A_GEN_WAV_DAC_POST_SNAKE, suffix=".beta"), b5)
        yield (self.format_tensor_name(T.A_GEN_WAV_DAC_POST_CONV, suffix=".weight"), get("decoder.decoder.6.conv.weight"))
        yield (self.format_tensor_name(T.A_GEN_WAV_DAC_POST_CONV, suffix=".bias"), get("decoder.decoder.6.conv.bias"))
