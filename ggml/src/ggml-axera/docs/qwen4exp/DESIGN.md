# Qwen3.8-Flash-Next / AX8860 设计

## 结论

- 官方结构以 Hugging Face Transformers `main` 的
  `qwen4_exp/{configuration,modeling}_qwen4_exp.py` 为准，commit
  `44ee1da4158b5cdb5f836d2ddc489d191bf7d49c`。llama.cpp `axera-dev` 已有完整
  `QWEN4EXP` CPU 图、GGUF 转换器和 MTP 图，可直接作为 CPU 真值实现。
- 48 层主干是 36 层 GDN 线性注意力加 12 层 QSA 全注意力，比例 3:1。
  0-based 全注意力层为 `3,7,...,47`。每层顺序为
  `PLE(layer 1 only) -> HC read -> attention -> HC write -> HC read -> MoE -> HC write`。
- **128 GB 是统一内存（用户 11-27 确认，CMM 与 Linux 共用）**。按实际
  NVFP4/Q8 GGUF 的 tensor 尺寸重算（`tools/cmm_budget.py`，日志
  `logs/cmm_budget_q38fn_v2.log`）：专家 NVFP4 `64.600 GiB`、文本 dense `8.325 GiB`，
  合计 `72.925 GiB` 权重；加 `-c 262144` 的 KV/indexer/GDN `7.046 GiB` 和
  `8 GiB` 工作区，文本 CMM 约 `87.971 GiB`，128 GiB 统一内存下余 `40.029 GiB`。
  默认 256K 可以把上下文降到 128K/32K 换余量（见下表）。
- **ngram 表不能常驻**：Q8_0 `50.66 GiB`、Q4_0 `26.82 GiB`、BF16
  `95.37 GiB`。256K 下 Linux 只剩 `40.03 GiB`，Q8_0 放不下，Q4_0 放得下。
  用户定的方案是 Q8_0 GGUF 放磁盘、`mmap` 只读按需访问（热行由 page cache
  自然缓存），host 每 token 多线程并发 gather 16 行；Q4_0/Q4_K 作为 PPL
  容差内的备选。kernel 只收 gather 后的 16×160 embedding。
- 上游最新 Flash-Next NPU 代码只有 QSA core 与 gated residual 原语，没有整模型
  kernel。作者分支 `origin/feat/qwen38-next-deepseek-v41` commit
  `16ba23039ae7e551d8b9d5e1da208202c51dce21`；ngram CPU op 分支
  `origin/feat/cpu-op-n-gram` commit
  `5fa9592fb0c13e3bb9f16abfada658a0349956a1`。

## 配置与权重

权重目录 `/data/junda/hf/Qwen3.8-Flash-Next`，131 个 safetensors 分片，
1658 个张量。

| 项目 | 数值 |
| --- | ---: |
| 总参数 | 179,999,981,459 |
| BF16 权重 | 359,999,963,128 B = 335.29 GiB |
| 路由专家 | 120,795,955,200 |
| ngram embedding | 51,200,245,760 |
| 其它（含视觉和 MTP） | 8,003,780,499 |

统计日志：`logs/weights.txt`、`logs/weights.json`。

关键配置：

| 参数 | 值 |
| --- | ---: |
| hidden | 2560 |
| layers | 48 |
| HC streams / low rank | 4 / 320 |
| routed experts / top-k | 512 / 10 |
| expert intermediate / shared intermediate | 640 / 640 |
| QSA Q heads / KV heads / head dim | 24 / 2 / 256 |
| partial RoPE | 0.25 -> 64 lanes |
| mRoPE sections / theta | [11,11,10] / 1e7 |
| indexer Q heads / KV heads / head dim | 4 / 1 / 128 |
| indexer budget / compress ratio | 2048 / 4 |
| GDN K heads / V heads / head dim | 16 / 48 / 128 |
| GDN conv kernel / chunk | 4 / 64 |
| PLE layer id | config 1-based [2] -> 0-based layer 1 |
| ngram size / heads per ngram | 3 / 8 -> 16 heads |
| ngram head dim / table row dim | 160 / 2560 |
| PLE conv kernel / dilation / history | 4 / 3 / 9 |
| vision | Qwen3-VL ViT, depth 27, hidden 1152, patch 16 |
| MTP | 1 full-QSA layer |

## 数学与执行顺序

所有 `Qwen4ExpTextRMSNorm` 的权重是零中心参数，增益为 `1 + w`。这包括
attention q/k norm、indexer q/k norm、PLE 的三个 grouped norm、HC norm 和
最终 HC read norm。GDN 的 `RMSNormGated.weight` 是普通 ones 初始化，不是
`1 + w`。

### 主干与 HC

初始 embedding `e[t,h]` 复制成四条 residual stream：

```text
R[t,c,h] = e[t,h], c=0..3
```

一次 HC read：

```text
n[t,c,h] = RMSNorm(R[t,c,:]; gain=1+w_hc[c,h])       # 每个 c 独立对 h 归约
F[t,:] = concat_c(n[t,c,:])                           # 10240
u[t,:] = SiLU(Linear(F[t,:], W_down) / 4)             # 320
g[t,c,h] = sigmoid(Linear(u[t,:], W_up)[c,h])
x[t,h] = sum_c(g[t,c,h] * n[t,c,h]) / 4
s[t,c] = 2 * sigmoid(Linear(F[t,:], W_inject)[c] / 4)
```

分支结束后：

```text
R'[t,c,h] = R[t,c,h] + s[t,c] * y[t,h]
```

`W_down` 为 `[320,10240]`，`W_up` 为 `[10240,320]`，
`W_inject` 为 `[4,10240]`。attention 和 MoE 各有独立的一组 HC 权重。
最后一层之后的 `hyper_connection_mixer` 只做 read，不生成 inject，然后直接进
lm_head；不得额外插普通 final RMSNorm。

### QSA 全注意力与 indexer

QSA 使用 token 级 BF16 paged K/V，cache 逻辑形状
`[pages,2,page_size=256,Hkv=2,256]`。QSA core 的 ABI 来自上游
`sgl-kernel-axera/python/sgl_kernel_axera/attention/qsa.py`：

```text
q:[T,24,256], cache:[P,2,256,2,256], selected:[T,K]
score = q @ gather(cache[page,0,offset,kv_head,:]).T / sqrt(256)
p = softmax(score over valid selected tokens)
o = p @ gather(cache[page,1,offset,kv_head,:])
```

Q 和 K 先做 per-head RMSNorm（gain `1+w`），再对最后 64 维做 interleaved
mRoPE。`q_proj` 输出 `24*256*2 = 12288`，前 6144 是 Q，后 6144 是 gate：

```text
attn_out = o * sigmoid(gate)
out = o_proj(attn_out)
```

indexer 的每层状态每个 cell 保存两份 128 维 BF16 key：未旋转 raw key 和块池化
key，因此每 cell 为 `2*128*2 = 512 B`。对每个 query：

```text
visible = 因果可见 token，按 cache 顺序排列
full_blocks = floor(len(visible) / 4)
raw = indexer_k(cell)
pooled[b] = mean(raw[visible[4b:4b+4]])              # FP32 均值后舍入 BF16
pooled[b] = RMSNorm(pooled[b]; gain=1+w)
pooled[b] = RoPE(pooled[b], position=visible[4b])
score[b] = sum_h relu(q[h] dot pooled[b]) / sqrt(128)
selected_blocks = topK(score, min(512, full_blocks))
selected_tokens = flatten(visible[selected_blocks*4 : selected_blocks*4+4])
selected_tokens += visible[full_blocks*4:]            # 不完整 tail 全部保留
```

`K` 的最大值为 `indexer_budget + compress_ratio - 1 = 2051`。随后 QSA core
对选中的原始 token 做普通 softmax attention，不使用池化 key 的数值。重复位置
保留重复次数。decode 时同样执行 indexer；不能退化成全上下文 dense attention。

### GDN 线性注意力

每层输入 `x` 经过四个投影：

```text
qkv = W_qkv x       # 10240 = 2048 q + 2048 k + 6144 v
z   = W_z x         # 6144
b   = W_b x         # 48
a   = W_a x         # 48
qkv = SiLU(causal_depthwise_conv1d(qkv; kernel=4))
```

`q,k` reshape 为 16 个 128 维 head，`v` 为 48 个 128 维 head。`q/k` 做 L2
norm，`q` 再乘 `1/sqrt(128)`。`q/k` 按连续 3 个 V head 复制。

```text
beta[vh] = sigmoid(b[vh])
g[vh] = -exp(A_log[vh]) * softplus(a[vh] + dt_bias[vh])
S[vh,kh,kv] = exp(g[vh]) * S[vh,kh,kv]
mem[vh,kv] = sum_kh S[vh,kh,kv] * k[vh,kh]
delta[vh,kv] = beta[vh] * (v[vh,kv] - mem[vh,kv])
S[vh,kh,kv] += k[vh,kh] * delta[vh,kv]
o[vh,kv] = sum_kh S[vh,kh,kv] * q[vh,kh]
```

状态 S 是 `[36 layers,48,128,128]` FP32。输出按 head 做 RMSNorm，乘普通
`ssm_norm` 权重，再乘 `sigmoid(z)`；`output_gate_type=sigmoid`，不是 qwen35
的 SiLU gate。最后 `out_proj` 回 2560。prefill 可使用上游 chunked gated delta
算法，但必须与上述 recurrent 公式在 teacher-forced 输入上对齐。

AX8860 实现约束：BF16 投影的 OC tile 必须是 128。512 时 layer0 融合图的 GDN
只有 `cos=0.8310636`，256 时 `0.9228749`，128 时 `0.9999844`；原因是过宽的
runtime-bank 输出 tile 在 one-pass 调度里被后续 op 覆盖。fresh prefill 的首个
32-token chunk 跳过恒为零的 state product；续页 prefill 的 `prefix > 0` 路径
仍需单独验收。

### PLE n-gram embedding

PLE 只在 0-based layer 1 执行。历史上下文长度为 2，初始补 EOS；前序出现过
EOS 时，跨 EOS 的窗口按参考实现回退 EOS。视觉位置使用
`image_token_id=248056` 顶替 token id。

```text
history[t] = [prev2, prev1, current]
for shift s in 0..2:
    shifted[s,t] = history[t-s] if s <= position_in_segment else EOS
```

对 `ngram=2` 和 `ngram=3` 分别计算 8 个 head：

```text
mixed[t] = XOR_{p=0..ngram-1} uint64(shifted[p,t] * layer_multiplier[p])
row[h,t] = mixed[t] % ngram_head_vocab_size[h] + ngram_head_offset[h]
emb[t,:] = concat_h(table[row[h,t], :])             # 16*160 = 2560
```

常量来自 checkpoint：`layer_multipliers[3]`、`ngram_heads_offsets[16]`、
`ngram_heads_vocab_sizes[16]`，均是精确 I64。源表有 128 个 shard，
逻辑形状为 `[51200245760,160]`。

PLE 投影和门控：

```text
key   = grouped_RMSNorm(W_key emb;   gain=1+w, group=2560)   # [4,2560]
value = W_value emb                                           # [2560]
query = grouped_RMSNorm(hidden;      gain=1+w, group=2560)   # [4,2560]
d[c]  = sum_h key[c,h] * query[c,h]
gate[c] = sign(d[c]) * sqrt(abs(d[c]) / sqrt(2560))
gated[c,h] = sigmoid(gate[c]) * value[h]
n = grouped_RMSNorm(gated; gain=1+w, group=2560)             # [4,2560]
out = gated + SiLU(dilated_causal_conv1d(n; kernel=4, dilation=3))
hidden += out
```

### MoE

每条 stream 混合后的 `x[t,:]` 同时进入 shared expert 和 routed experts：

```text
logits = W_router x                                      # [512]
p = softmax(logits; FP32)
(w_i, e_i) = topK(p, 10); w_i /= sum(w_i)
routed = sum_i w_i * W_down[e_i] (SiLU(W_gate[e_i] x) * W_up[e_i] x)
shared = W_down_s (SiLU(W_gate_s x) * W_up_s x)
out = routed + sigmoid(W_shared_gate x) * shared
```

专家权重布局为 `gate_up_proj[512,1280,2560]` 和
`down_proj[512,2560,640]`。NVFP4 按 64 元素一块，36 B/块，即 4.5 bit/元素。

### MTP

MTP 是主干后的一个 full-QSA 层。权重名给出：

```text
mtp.fc_embedding.weight [2560,2560]
mtp.fc_hidden.weight    [2560,10240]
mtp.pre_fc_norm_embedding / pre_fc_norm_hidden
mtp.layers.*             一个 full-attention QSA block
mtp.hyper_connection_mixer
```

llama.cpp `src/models/qwen4exp.cpp` 将该层解释为
`eh_proj([enorm(e); hnorm(h)_s]) = fc_embedding(e) + fc_hidden(h_s)`，四个 HC
stream 共享该投影，随后复用完整 QSA block，最后 HC read + 共享 lm_head。
Transformers 主仓没有 MTP 训练侧实现；本适配的 MTP 真值暂以 llama.cpp 图和
权重名逐层对齐为准，不据此猜其他未覆盖结构。

### Vision

视觉塔是未修改的 Qwen3-VL ViT：depth 27、hidden 1152、16 heads、
intermediate 4304、patch 16、temporal patch 2、spatial merge 2、输出 2560。
一个 ViT axmodel，不按图像数量拆 kernel。

## 状态与 KV 布局

以下按最大 `-c 262144` 计算。QSA 页大小 256，每层 1024 页。

| 状态 | 布局 | 大小 |
| --- | --- | ---: |
| 12 层 QSA K/V | 每层 `[1024,2,256,2,256]` BF16 | 6.00 GiB |
| 12 层 indexer | 每层每 cell raw+pooled `128*2*2 B` | 1.50 GiB |
| MTP QSA K/V | `[1024,2,256,2,256]` BF16 | 0.50 GiB |
| MTP indexer | 每 cell raw+pooled BF16 | 0.125 GiB |
| 36 层 GDN recurrent | 每层 `[48,128,128]` FP32 | 108 MiB |
| 36 层 GDN conv | 每层 `[10240,3]`，按 FP32 保守计 | 4.2 MiB |
| PLE conv | `[10240,9]`，按 FP32 计 | 0.35 MiB |
| PLE 历史 | 2 个 I32 token | 8 B |

QSA K/V 直接绑定 kernel IO，按页写入，不做 host/CMM 往返。indexer 的 raw 和
pooled 行可在同一 cell 内连续布局。GDN 状态跨请求保存；PLE 历史、conv 状态和
位置也必须进入 seqstate 快照。

## CMM 与 host 内存预算

### 文本 CMM

| 项目 | 大小 |
| --- | ---: |
| 路由专家 NVFP4（48 层 gate_up + down，含 E4M3 block scale） | 64.600 GiB |
| 文本 dense（attention/MoE shared/PLE/HC/lm_head；不含 host embedding） | 8.325 GiB |
| **文本权重小计（CMM bank）** | **72.925 GiB** |

数值来自实际 `gguf/q38fn-nvfp4-q8.gguf` 的每张量字节数
（`tools/cmm_budget_q38fn.py` → `logs/cmm_budget_q38fn_v2.log`）。NVFP4 不是 4 bit：
GGML 每 64 个值 36 B（32 B packed + 4 B E4M3 scale），即 4.5 bit；按 4 bit
会把专家少算约 7 GiB。状态与工作区按 `-c` 缩放（QSA 页 256；本轮最终包不含
MTP/ViT）：

| 项目 | 32K | 128K | 256K |
| --- | ---: | ---: | ---: |
| QSA K/V (BF16, 12 层) | 0.750 GiB | 3.000 GiB | 6.000 GiB |
| indexer raw+pooled (12 层) | 0.117 GiB | 0.469 GiB | 0.938 GiB |
| GDN recurrent+conv (36 层) | 0.108 GiB | 0.108 GiB | 0.108 GiB |
| kernel 工作区预留 | 8.00 GiB | 8.00 GiB | 8.00 GiB |
| **CMM 合计** | **81.900 GiB** | **84.502 GiB** | **87.971 GiB** |
| 统一内存下 Linux 余量 | 46.100 GiB | 43.498 GiB | 40.029 GiB |

视觉塔（0.45 B，BF16 约 0.84 GiB）单独放在 `q38fn_vit`，只在有图时加载，
不与文本常驻权重相加。上表给 bank padding、运行时双缓冲和框架开销留了余量；
最终以实际 axmodel 的 weight bank 清单为准。

### host ngram 表

| 精度 | 大小 |
| --- | ---: |
| Q8_0 | 50.664 GiB |
| Q4_0 | 26.822 GiB |
| BF16 | 95.368 GiB |

Q8_0 每行 160 个元素，按 32 元素块为 34 B，即 170 B/row。每次 token gather
16 行，约 2.7 KiB/token；prefill 256 行约 696 KiB/gather 批次。host 到 NPU
每 token 送 `16*160` 个反量化值；若送 BF16 为 5 KiB/token。

统一内存结论：256K 下 CMM `87.971 GiB` 后 Linux 只剩 `40.029 GiB`，Q8_0
（`50.66 GiB`）不能常驻；Q4_0（`26.82 GiB`）可以常驻。因此按用户定的方案：
Q8_0 表放磁盘 `mmap` 按需读（热行由 page cache 缓存），host 每 token
多线程 gather 16 行后把 5 KiB BF16 送给 NPU；同时在 CPU 上评估 Q4_0/Q4_K
的 PPL，若差距在容差内且磁盘读太慢则改用 Q4_0 常驻。不做专家搬运，不剪专家。

## Kernel 清单草案

### 本轮实际编出的 axmodel（2026-11-28）

| 文件 | 大小 | 说明 |
| --- | ---: | --- |
| `decode_ctx4096.axmodel` | 5,811,237 B | 4 核 decode T=1，48 层运行时循环，`lm_head` 内联 |
| `decode_ctx262144.axmodel` | 5,876,013 B | 同上，256K 状态 |
| `prefill_ctx256.axmodel` | 86,557,345 B | 4 核 prefill T=256，运行时 12 block 循环 |

- 路由专家按用户 11-28 确认不做 4 GiB 切块：每层 `moe_expert_gate_up` /
  `moe_expert_down` 是一个 NVFP4 runtime bank IO；编译和 CRUN 未报单 IO 限制。
- prefill QSA 以 64 行 tile 跑 256 行（`q_all/selected/attention` 三块合计
  约 8 MiB，超过单 job OCM 预算）；HC 把 `[256,4,2560]` 按 8 行 tile 分 32 次
  搬入 OCM 计算并分块写回。整块残差 buffer 留在 DDR，避免 5 MiB 常驻与 QSA/
  MoE 工作区争用。MoE/PLE 由各自 kernel 内部分块。
- decode 的 HC read 结果 materialize 到 DDR，再由 GDN/QSA/MoE 的 published
  input 路径消费。`oc_tile_width=128` 修复后整层 CRUN 已过；旧
  `HC read -> GDN` 失败记录保留在根目录 `SOLVED_gdn_handoff.md`。

最终发布包只保留三个 axmodel，全部四核。

| 文件 | 职责 | 每次推理调用 |
| --- | --- | ---: |
| `q38fn_prefill.axmodel` | 48 层文本 prefill，attention 内层 256 行，MoE 4096 行，动态 mask/indexer/tail | 每 prefill chunk 1 次 |
| `q38fn_decode.axmodel` | 48 层 T=1 decode，GDN/QSA/indexer/PLE/HC/MoE，lm_head argmax 与 top-k 采样并入 | 每 token 1 次 |
| `q38fn_vit.axmodel` | Qwen3-VL ViT 与 2x2 merge，一个模型覆盖所有图像 | 每张图 1 次 |

算子来源与组装：

| 算子 | 实现来源 |
| --- | --- |
| QSA sparse softmax/AV | 上游 `attention/qsa.py::qsa_attention`，不复制不改 |
| HC read/write | 上游 `models/common/gated_residual.py` |
| GDN | 上游 qwen35/35B gated delta 原函数 |
| MoE prefill/decode | `models/moe_gqa`、qwen36 35B 的 expert-parallel MoE 与 prefill_small 路径 |
| dense GEMM/norm/rope/top-k | 上游现有 linear、RMSNorm、RoPE、top-k、gather 原语 |
| indexer | 投影、均值池化、RMSNorm、RoPE、matmul、relu、sum、top-k、gather 组装 |
| PLE gather | host 做 hash + Q8 gather，NPU 收 embedding；NPU 侧投影/门控/conv 用现有原语组装 |
| ViT | 现成 Qwen3-VL 合并 kernel |

构建脚本只提供真实形状、bank、IO 名并直接调用上游函数。不能为某个 prompt
长度新增变体，不能按名字 if，不能带 CPU fallback。

### indexer kernel 实现约束（CRUN 已验证）

实现见 `models/qwen38_fn/indexer.py`，测试 `tests/test_qwen38_fn_indexer.py`。
这些约束来自真实 CRUN 踩坑，整模型 decode 复用同一 kernel 时必须遵守：

- **可组合入口**：主体是普通函数 `indexer_select_value(...)`，整模型 builder
  直接内联；`indexer_select_kernel` 只是 standalone 包装（调用后
  `return selected`）。`indexer_qk` 也是普通函数，投影/norm/RoPE 可直接组合。

- **多核行分片**：`rows >= cores` 时 balanced split；`rows < cores`（T=1
  decode）时**四个核都计算同一行**（`row_start=0,row_stop=rows`），重复写
  的 raw/selected 是同值，竞争无害，BlockGraph 完全一致。已去掉原来的
  `rows >= cores` 断言。CRUN `18 passed`（`logs/q38_fn_indexer_crun_v25_multicore.log`），
  4 核 T=1 可直接内联进 decode。
- **每核 BlockGraph 必须一致**：所有核都要执行同样的 pooling 循环和
  `dynamic_st`，不能按 `core` 静态跳过某核；只允许运行期数据不同。
- **整数 floor 除法一律 `high_precision=True`**：CRUN 的 reciprocal 快捷路径
  会把 `8/4`、`2/4` 算错，影响 block 数和 token 展开。
- **pooling 写入下限 1 行**：decode 常见某步没有完成任何新 block，
  `dynamic_st` 的 copy 长度会为 0（PYRUN 断言失败、硬件也是空 DMA）。
  把 store 行数 clamp 到 `max(count,1)`；多出的行落在 `>= blocks` 的未读区，
  等该 block 填满时被覆盖。
- **cache/rotary 需留 padded 余量**：`cos_block/sin_block` 每个核一行按
  `cores * ceil(ceil(rows/4)/cores)` padded block 给；`raw_cache` 多留
  `4*padded` 行、`pooled_cache` 多留 `padded` 行、RoPE 表多留
  `4*padded` 行。生产上均按 capacity 尺寸，天然满足。
- **token 展开按 64 行分块**：避免 `[256,2048]` 的 expanded/tokens 同时驻留
  OCM；`top_ids` padding 用 `-1` 并显式判 `0 <= id < blocks`。
- **Induction tensor 不复用常量**：`ax.dsl.range` 的第一个参数会被原地递增，
  不能拿 `zero/start` 之类常量当循环下标。

日志：PYRUN `logs/q38_fn_indexer_pyrun_final5.log`（15 passed, 3 skipped），
CRUN `logs/q38_fn_indexer_crun_v23.log`（21 passed, 3 skipped，含 PLE）。

## 验证门禁

1. 上游 `tests/test_qsa.py` 与 `tests/test_gated_residual.py` 必须 CRUN 通过。
2. 固定 prompt 的逐层 hidden 与 logits 对 Transformers 参考；
   teacher-forced 每层 cos 目标不低于 0.999。
3. 每个 NPU kernel 用真实权重和真实输入跑 CRUN，再对 CPU 参考；PYRUN 不算。
4. 短 PPL 使用 `ppl_zh.txt` 拷贝，`-c 512 --chunks 3`。
5. 板子到手前不连接任何板子。板上测试由用户给出地址和纪律后再执行。

## 未决与风险

- 统一内存下 Q8_0 ngram 表（50.66 GiB）与 256K 文本 CMM 不能同时常驻；
  方案是 Q8_0 `mmap` 按需读 + Q4_0 备选，需在板上实测 gather 冷/热延迟。
- Transformers 主仓没有 MTP 训练侧代码；MTP 真值需以 llama.cpp 图逐张量对齐，
  并在报告中明确其来源边界。
- NVFP4 权重的最终 CMM bank 大小要按实际打包后重算；本文的 63.281 GiB 是
  4.5 bit/元素的算术下界。
- 全模型 QSA/indexer/PLE/MoE 合并 kernel 尚需实际编译和 CRUN，当前只完成上游
  QSA 与 HC 原语验证。
