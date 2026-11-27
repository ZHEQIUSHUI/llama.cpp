from __future__ import annotations

import re
from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, LazyTorchTensor

# E2M1 values indexed by the 4-bit code (low nibble = even element, high = odd),
# matching the official Inkling2LoopDequantizeExperts table.
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

ARCH = "inkling2_loop"


@ModelBase.register("Inkling2LoopForConditionalGeneration", "Inkling2LoopForCausalLM")
class Inkling2LoopModel(TextModel):
    model_arch = gguf.MODEL_ARCH.INKLING2_LOOP

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_layer = int(self.hparams["num_hidden_layers"])
        loop = self.hparams.get("loop_config") or {}
        self.loop_start = int(loop.get("core_start", self.hparams.get("loop_layer_index_start", 6)))
        self.loop_end = int(loop.get("core_end", self.hparams.get("loop_layer_index_end", 18)))
        self.loop_passes = int(loop.get("num_passes", self.hparams.get("loop_num_passes", 3)))
        # the GGUF block count is the expanded number of cache slots; the looped
        # layers reuse one weight bank and alias the loader's layer pointers.
        self.n_slot = self.n_layer + (self.loop_passes - 1) * (self.loop_end - self.loop_start)
        self.block_count = self.n_slot
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)
        self._done_expert_layers: set[int] = set()
        # the checkpoint pads embed/unembed to vocab_size=201024 but the real
        # (unpadded) vocabulary and lm_head are 200058 tokens; keep the GGUF
        # vocabulary unpadded so tokenizer n_vocab matches the output head.
        self.hparams["vocab_size"] = int(self.hparams.get("unpadded_vocab_size", self.hparams["vocab_size"]))

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        h = self.hparams
        tc = h  # text_config fields are flattened into self.hparams by TextModel
        gw = self.gguf_writer

        def kv_u32(key, val):
            gw.add_key_value(f"{ARCH}.{key}", int(val), gguf.GGUFValueType.UINT32)

        def kv_f32(key, val):
            gw.add_key_value(f"{ARCH}.{key}", float(val), gguf.GGUFValueType.FLOAT32)

        loop = h.get("loop_config") or {}
        self.loop_start = int(loop.get("core_start", h.get("loop_layer_index_start", 6)))
        self.loop_end = int(loop.get("core_end", h.get("loop_layer_index_end", 18)))
        self.loop_passes = int(loop.get("num_passes", h.get("loop_num_passes", 3)))
        kv_u32("loop.layer_start", self.loop_start)
        kv_u32("loop.layer_end", self.loop_end)
        kv_u32("loop.passes", self.loop_passes)
        kv_u32("loop.gate_rank", int(loop.get("gate_rank", h.get("loop_gate_rank", 64))))

        kv_u32("attn_rel.dim", int(h.get("d_rel", 16)))
        kv_u32("attn_rel.extent", int(h.get("rel_extent", 1024)))
        kv_f32("attn_rel.clip", float(h.get("rel_l2_clip_threshold", 8.0)))
        kv_f32("attn_qk.clip", float(h.get("qk_l2_clip_threshold", 22.627416997969522)))
        kv_u32("log_scale.floor", int(h.get("log_scaling_n_floor", 8192)))
        kv_f32("log_scale.alpha", float(h.get("log_scaling_alpha", 0.1)))
        kv_f32("mlp_gate_softcap", float(h.get("mlp_gate_softcap", 4.0)))
        kv_f32("mlp_up_softcap", float(h.get("mlp_up_softcap", 25.0)))
        kv_u32("sconv.kernel", int(h.get("conv_kernel_size", h.get("sconv_kernel_size", 4))))
        kv_u32("attn_res.block", int(h.get("n_layers_per_attn_res_block", 6)))
        local_ids = list(h.get("local_layer_ids", []))
        gw.add_key_value(f"{ARCH}.local_layers", local_ids, gguf.GGUFValueType.ARRAY,
                         sub_type=gguf.GGUFValueType.UINT32)

        # MoE metadata
        gw.add_expert_feed_forward_length(int(h.get("intermediate_size", h.get("moe_intermediate_size", 1152))))
        gw.add_expert_weights_scale(float(h.get("route_scale", 8.0)))
        gw.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        gw.add_head_count_kv(int(h.get("num_key_value_heads", 8)))
        gw.add_key_length(int(h.get("head_dim", 128)))
        gw.add_value_length(int(h.get("head_dim", 128)))
        gw.add_sliding_window(int(h.get("sliding_window_size", 1024)))
        if (sc := h.get("final_logit_softcapping")) is not None:
            gw.add_final_logit_softcapping(float(sc))

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        f32 = ("attn_rel_linear", "attn_rel_proj", "attn_rel_scale", "attn_q_scale",
               "attn_sink_proj", "attn_q_input_norm", "attn_k_input_norm", "attn_v_input_norm",
               "attn_conv", "mlp_conv", "k_conv", "v_conv", "k_conv_static", "v_conv_static",
               "k_conv_proj", "v_conv_proj", "attn_input_q", "mlp_input_q", "msg_down",
               "msg_forget", "msg_gain", "attn_res_final_q", "message_input_q", "embed_norm")
        if any(k in new_name for k in f32):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    # ---- experts ---------------------------------------------------------
    def _get(self, name: str) -> Tensor:
        gen = self.model_tensors[name]
        t = gen()
        if isinstance(t, list):
            t = t[0]
        return t

    def _dequant_experts_layer(self, bid: int) -> tuple[Tensor, Tensor]:
        p = f"model.llm.layers.{bid}.mlp.experts."
        eager = LazyTorchTensor.to_eager
        w13 = eager(self._get(p + "w13_weight"))
        s13 = eager(self._get(p + "w13_weight_scale"))
        w2 = eager(self._get(p + "w2_weight"))
        s2 = eager(self._get(p + "w2_weight_scale"))

        nvfp4 = (p + "w13_weight_amax") in self.model_tensors
        if nvfp4:
            a13 = eager(self._get(p + "w13_weight_amax")).float()
            a2 = eager(self._get(p + "w2_weight_amax")).float()
            den = float(eager(self._get(p + "w13_weight_outer_scale_denominator")).float().reshape(-1)[0])

        e2m1 = torch.tensor(_E2M1, dtype=torch.float32).numpy()
        n_e = w13.shape[0]
        gu_list, dn_list = [], []
        for e in range(n_e):
            if nvfp4:
                c13 = w13[e].to(torch.uint8).cpu().long()
                v13 = torch.empty(c13.shape[0], c13.shape[1] * 2, dtype=torch.float32)
                v13[:, 0::2] = torch.from_numpy(e2m1[(c13 & 0xF).numpy()])
                v13[:, 1::2] = torch.from_numpy(e2m1[(c13 >> 4).numpy()])
                v13 *= s13[e].float().cpu().repeat_interleave(16, dim=1) * (a13[e] / den)

                c2 = w2[e].to(torch.uint8).cpu().long()
                v2 = torch.empty(c2.shape[0], c2.shape[1] * 2, dtype=torch.float32)
                v2[:, 0::2] = torch.from_numpy(e2m1[(c2 & 0xF).numpy()])
                v2[:, 1::2] = torch.from_numpy(e2m1[(c2 >> 4).numpy()])
                v2 *= s2[e].float().cpu().repeat_interleave(16, dim=1) * (a2[e] / den)
            else:
                v13 = w13[e].float().cpu() * torch.exp2(s13[e].float().cpu() - 127).repeat_interleave(32, dim=1)
                v2 = w2[e].float().cpu() * torch.exp2(s2[e].float().cpu() - 127).repeat_interleave(32, dim=1)

            # w13 source rows are [gate0, up0, gate1, up1, ...] -> [gate ; up]
            gate = v13[0::2]
            up = v13[1::2]
            # keep f32 here; prepare_tensors cannot call .numpy() on bf16 tensors
            gu_list.append(torch.cat([gate, up], dim=0).contiguous())
            dn_list.append(v2.contiguous())

        return torch.stack(gu_list, dim=0), torch.stack(dn_list, dim=0)

    # ---- tensor mapping --------------------------------------------------
    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # text tower only; vision handled by the mmproj converter
        if name.startswith("model.visual."):
            return
        if name.startswith("model.llm.mtp.") or name.startswith("mtp."):
            return

        if name == "model.llm.embed.weight":
            n_unpadded = int(self.hparams.get("unpadded_vocab_size", data_torch.shape[0]))
            yield self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD), data_torch[:n_unpadded].contiguous()
            return
        if name == "model.llm.embed_norm.weight":
            yield self.format_tensor_name(gguf.MODEL_TENSOR.INKLING_EMBED_NORM), data_torch
            return
        if name == "model.llm.unembed.weight":
            n_unpadded = int(self.hparams.get("unpadded_vocab_size", data_torch.shape[0]))
            yield self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT), data_torch[:n_unpadded].contiguous()
            return
        if name == "model.llm.norm.weight":
            yield self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT_NORM), data_torch
            return
        if name == "model.llm.attn_res.output_query":
            yield self.format_tensor_name(gguf.MODEL_TENSOR.INKLING_FINAL_Q), data_torch
            return
        m = re.fullmatch(r"model\.llm\.attn_res\.attn_queries\.(\d+)", name)
        if m:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.INKLING_ATTN_INPUT_Q, int(m.group(1)) + 1), data_torch
            return
        m = re.fullmatch(r"model\.llm\.attn_res\.mlp_queries\.(\d+)", name)
        if m:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.INKLING_MLP_INPUT_Q, int(m.group(1))), data_torch
            return
        if name == "model.llm.middle_cycle_cross_pass_queries":
            self._msg_q = getattr(self, "_msg_q", {})
            self._msg_q["q"] = data_torch
            if "kv" in self._msg_q:
                yield self._merged_msg_q()
            return
        if name == "model.llm.middle_cycle_cross_pass_kv_queries":
            self._msg_q = getattr(self, "_msg_q", {})
            self._msg_q["kv"] = data_torch
            if "q" in self._msg_q:
                yield self._merged_msg_q()
            return

        # experts: dequantize the whole layer once
        m = re.fullmatch(r"model\.llm\.layers\.(\d+)\.mlp\.experts\.(w13_weight|w2_weight)(?:_(?:scale|amax|outer_scale_denominator))?", name)
        if m:
            layer = int(m.group(1))
            if layer in self._done_expert_layers:
                return
            self._done_expert_layers.add(layer)
            gu, dn = self._dequant_experts_layer(layer)
            yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_UP_EXP, layer), gu
            yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_DOWN_EXP, layer), dn
            return

        m = re.fullmatch(r"model\.llm\.layers\.(\d+)\.(.+)", name)
        if not m:
            return
        layer = int(m.group(1))
        rest = m.group(2)

        simple = {
            "attn.wq_du.weight": gguf.MODEL_TENSOR.ATTN_Q,
            "attn.wk_dv.weight": gguf.MODEL_TENSOR.ATTN_K,
            "attn.wv_dv.weight": gguf.MODEL_TENSOR.ATTN_V,
            "attn.wo_ud.weight": gguf.MODEL_TENSOR.ATTN_OUT,
            "attn.gate_attn.weight": gguf.MODEL_TENSOR.ATTN_GATE,
            "attn.wr_du.weight": gguf.MODEL_TENSOR.INKLING_ATTN_REL_LINEAR,
            "attn.rel_logits_proj.proj": gguf.MODEL_TENSOR.INKLING_ATTN_REL_PROJ,
            "attn.rel_l2_clip.scale": gguf.MODEL_TENSOR.INKLING_ATTN_REL_SCALE,
            "attn.q_l2_clip.scale": gguf.MODEL_TENSOR.INKLING_ATTN_Q_SCALE,
            "attn.attn_sink_proj.weight": gguf.MODEL_TENSOR.INKLING_ATTN_SINK_PROJ,
            "attn_norm.weight": gguf.MODEL_TENSOR.ATTN_NORM,
            "q_input_norm.weight": gguf.MODEL_TENSOR.INKLING_ATTN_Q_NORM,
            "k_input_norm.weight": gguf.MODEL_TENSOR.INKLING_ATTN_K_NORM,
            "v_input_norm.weight": gguf.MODEL_TENSOR.INKLING_ATTN_V_NORM,
            "attn_sconv.weight": gguf.MODEL_TENSOR.INKLING_ATTN_CONV,
            "mlp_sconv.weight": gguf.MODEL_TENSOR.INKLING_MLP_CONV,
            "attn.k_sconv.weight": gguf.MODEL_TENSOR.INKLING_K_CONV,
            "attn.v_sconv.weight": gguf.MODEL_TENSOR.INKLING_V_CONV,
            "attn.k_sconv.static_w": gguf.MODEL_TENSOR.INKLING_K_CONV_STATIC,
            "attn.v_sconv.static_w": gguf.MODEL_TENSOR.INKLING_V_CONV_STATIC,
            "attn.k_sconv.weight_proj": gguf.MODEL_TENSOR.INKLING_K_CONV_PROJ,
            "attn.v_sconv.weight_proj": gguf.MODEL_TENSOR.INKLING_V_CONV_PROJ,
            "mlp_norm.weight": gguf.MODEL_TENSOR.FFN_NORM,
            "mlp.gate.weight": gguf.MODEL_TENSOR.FFN_GATE_INP,
            "mlp.gate.bias": gguf.MODEL_TENSOR.FFN_EXP_PROBS_B,
        }
        if rest in simple:
            t = data_torch
            if rest in ("attn_sconv.weight", "mlp_sconv.weight",
                        "attn.k_sconv.weight", "attn.v_sconv.weight"):
                t = t.squeeze(1)                       # [C,1,k] -> [C,k]  (cc, conv)
            elif rest in ("attn.k_sconv.static_w", "attn.v_sconv.static_w"):
                t = t.T.contiguous()                   # [k,C] -> [C,k]    (cc, conv)
            elif rest == "attn.rel_logits_proj.proj":
                t = t.T.contiguous()                   # [d_rel,extent] -> [extent,d_rel] (matmul op)
                ext = int(self.hparams.get("rel_extent", t.shape[0]))
                if t.shape[0] == ext:
                    pad = torch.zeros((ext + 1, t.shape[1]), dtype=t.dtype)
                    pad[:ext] = t
                    t = pad
            yield self.format_tensor_name(simple[rest], layer), t
            return

        m2 = re.fullmatch(r"controllers\.(attn|k|v|mlp)\.(down|forget|gain)_weight", rest)
        if m2:
            which = m2.group(1)
            kind = m2.group(2)
            # Four branch controllers (q/attn, k, v, mlp) are fused into one
            # row-concatenated blob per (layer, kind), in the fixed order
            # [attn, k, v, mlp]. The host/loader splits them back.
            gates = getattr(self, "_msg_gates", None)
            if gates is None:
                gates = self._msg_gates = {}
            slot = gates.setdefault((layer, kind), {})
            slot[which] = data_torch
            if len(slot) == 4:
                order = ["attn", "k", "v", "mlp"]
                fused = torch.cat([slot[b] for b in order], dim=0).contiguous()
                tensor = {"down": gguf.MODEL_TENSOR.INKLING_MSG_DOWN,
                          "forget": gguf.MODEL_TENSOR.INKLING_MSG_FORGET,
                          "gain": gguf.MODEL_TENSOR.INKLING_MSG_GAIN}[kind]
                yield self.format_tensor_name(tensor, layer), fused
            return

        raise ValueError(f"unhandled Inkling2Loop tensor: {name}")

    def _merged_msg_q(self):
        q = self._msg_q["q"]  # [24, H] -> [12,2,H] = [q, mlp]
        kv = self._msg_q["kv"]  # [24, H] -> [12,2,H] = [k, v]
        q = q.reshape(-1, 2, q.shape[-1])
        kv = kv.reshape(-1, 2, kv.shape[-1])
        merged = torch.cat([q[:, :1], kv, q[:, 1:]], dim=1)  # [12, 4, H]
        return self.format_tensor_name(gguf.MODEL_TENSOR.INKLING_MSG_INPUT_Q), merged.contiguous()
