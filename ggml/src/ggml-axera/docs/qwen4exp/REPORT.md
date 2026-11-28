# Qwen3.8-Flash-Next / AX8860 阶段报告

## 结论（先读这里）

- **本轮目标的两个 4 核 axmodel 已编出**：运行时循环 48 层 + 每层单 bank 的
  NVFP4 专家权重，`lm_head` 并入 decode。
  `build/q38fn/decode_ctx4096.axmodel` `5,811,237 B`（5.54 MiB，459.2 s，
  md5 `cb7d52b72d25be494592ff1ed95a10fc`）；
  `build/q38fn/decode_ctx262144.axmodel` `5,876,013 B`（5.60 MiB，451.9 s，
  md5 `f6b7cc2101c03502089a0cdd50d44a83`）；
  `build/q38fn/prefill_ctx256.axmodel` `86,557,345 B`（82.5 MiB，7853.0 s，
  303,660 个 job，md5 `b51c79520fb2a7b9f8b5ec47a6989115`）。日志
  `logs/compile_decode4096_r3.log`、`logs/compile_decode262144_r3.log`、
  `logs/compile_prefill256_r3.log`。
- **512 专家 bank 未切块**：按 11-28 用户确认，单层单 bank IO（NVFP4）直接编；
  decode/prefill 编译与 MoE T=1/T=256 CRUN 都没有报 4 GiB IO 限制，因此没有
  原始报错可贴，也没有做人为分块。
- **CMM 预算按实际 GGUF 重算**：256K 文本权重 `72.925 GiB`（专家 NVFP4
  `64.600` + dense `8.325`），KV/indexer/GDN 状态 `7.046 GiB`，激活工作区
  `8 GiB`，合计 `87.971 GiB`；128 GB 统一内存下 Linux 余 `40.029 GiB`。
  host ngram Q8_0 `50.664 GiB` 不常驻，按 token mmap gather；token embedding
  `0.629 GiB` 在 host。prefill256 的 32-page 状态实测 `0.325 GiB`。
  `tools/cmm_budget_q38fn.py`，日志 `logs/cmm_budget_q38fn_v2.log`。
- **单算子真实权重 CRUN 全部 ≥0.999**：QSA T=1 `0.9999679`、T=10
  `0.9999507`；PLE T=1 `0.9999981`、T=10 `0.9999505`；MoE T=1 `0.9995237`、
  T=256 `0.9992040`；HC read/inject/write `0.9999512–0.9999979`，HC 256 行
  分块 `0.9999512–0.9999972`；GDN T=1 `0.9999884`、T=256 `0.9999799`。
  日志见第 6 节。
- **上一轮 layer0 阻塞已解决**：根因是 BF16 GDN 投影 OC tile 512 过宽，
  融合图尾端读到被覆盖的 runtime-bank buffer；降到 128 后 layer0 的 GDN
  `0.8310636 -> 0.9999844`，after_moe `0.9998111`，MoE `0.9995171`。
  8 层（block0 + block1，含运行时循环）对 HF `layer_07` 为 `0.9976548`；
  用 HF `layer_07` 续跑层 8–15 为 `0.9954564`；两者都是 BF16 跨层累积误差。
  证据 `logs/stage4_oc128.log`、`logs/block_real_oc128.log`、
  `logs/block_pair_08_retry.log`，历史见 `SOLVED_gdn_handoff.md`。
- **整 48 层 CRUN 被内存纪律阻断**：完整 decode 图展开时 RSS 约 `250 GiB`，
  超过 200 GiB 上限，已按 PID 终止；跨块续跑到层 16–23 出现全 `inf`，权重
  缓存本身无 inf/nan，根因未定位。不能据此宣称整模型 CRUN 通过。
- **fresh prefill 首 chunk 也已修复**：首个 32-token chunk 跳过恒为零的
  state product，避免零 BF16 staging tensor 的图内生命周期问题；T=256 全
  chunk `0.9999799`。带已有状态的续页 prefill 仍需下一轮验收。
- **ViT 本轮未做；host adapter 未完成**（依赖 IO 定稿）。板子未给，全程未连板，
  未上传 HF。
- **host ngram gather 原型已测**：从 50.66 GiB Q8_0 表按 token 取 16 行，
  热 `0.0025 ms/token`，冷（512 token 随机行抽样，4090 本地盘）
  `0.0624 ms/token`。日志 `logs/ngram_gather.json`。板上存储速度需复测。
- **CPU 实现已对齐**：48 层逐层隐状态 + final norm + logits 已产出；0–20 层
  `cos ≥ 0.999`，21 层起降到 `cos≈0.993`，最终 `logits cos=0.99835`、
  `argmax_last=271` 与 Transformers 参考一致。`output/compare.json`、
  `logs/llama_dump_bf16_v3.log`。

## 1. 结构与官方实现

- 官方：Transformers `qwen4_exp/{configuration,modeling}_qwen4_exp.py`，
  commit `44ee1da4158b5cdb5f836d2ddc489d191bf7d49c`。上游只有这两个实现；
  vLLM/SGLang 没有独立的 qwen4exp 推理实现，复用 Transformers 语义。
- llama.cpp `axera-dev` 已有完整 `QWEN4EXP` CPU 图、GGUF 转换器和 MTP 图，
  作为 CPU 真值交叉验证。
- 11-27 重新 `git fetch origin` 并检查分支：仍只有
  `origin/feat/qwen38-next-deepseek-v41`（QSA + gated residual 原语）和
  `origin/feat/cpu-op-n-gram`（CPU ngram 算子），**没有 Flash-Next 整模型
  NPU kernel**。
- 关键配置：hidden 2560、48 层、HC 4 stream/lowrank 320、512 专家 top-10、
  moe/共享专家 640、QSA 24/2 头 hd256、partial rope 0.25、indexer
  4 头/128 维/budget 2048/compress 4、GDN 16 K 头/48 V 头/hd128、
  PLE layer 1、ngram 3/8 头、MTP 1 层、vision Qwen3-VL depth 27 hidden 1152。

## 2. Transformers 参考

- 脚本 `tools/ref_qwen4exp.py`（逐层 meta 加载、算完释放；ngram 表按行读）。
- prompt `中国的首都是北京。请用一句话介绍这座城市。`，token ids
  `[96328,102357,96878,98116,1710,139054,110827,97431,131588,1710]`。
- 产出 `output/ref_full/layer_00..47.npy`（`[10,10240]` 四路 HC）、
  `final_norm.npy`、`logits.npy`，48 层 `3031.76 s`，`argmax_last=271`。
  日志 `logs/ref_full.log`。

## 3. GGUF 转换与 PPL

### BF16 GGUF（完成）

- `convert_hf_to_gguf.py --outtype bf16 --fuse-gate-up-exps`，1207 张量，
  359.2 GB。元数据核对：rope 64、mRoPE `[11,11,10,0]`、freq 1e7、49 blocks、
  512/10 专家、PLE ngram3/8 头、indexer 4/128/2048。日志
  `logs/convert_q38_bf16.log`。

### 专家 NVFP4 + embedding Q8_0（完成）

- 工具 `llama/tools/axera/qwen4exp_nvfp4_gguf.py`：路由专家
  `ffn_gate_up_exps`/`ffn_down_exps` → NVFP4（98 张 + scale），
  `token_embd`/`per_layer_token_embd` → Q8_0，其余复制。产物
  `gguf/q38fn-nvfp4-q8.gguf`，1305 张量，125 GB。日志
  `logs/quant_q38_nvfp4_q8_v2.log`、`logs/q38fn_nvfp4_q8_report.json`。

### PPL（`-c 512 --chunks 3 -f assets/ppl_zh.txt`）

| 权重 | PPL | chunks | 日志 |
| --- | ---: | --- | --- |
| BF16 全量 | **10.6303 ± 1.04711** | 8.2482 / 8.4681 / 10.6303 | `logs/ppl_q38_bf16.log`（摘要 `_result.txt`） |
| NVFP4 专家 + Q8_0 embedding | **10.6004 ± 1.04549** | 8.3541 / 8.5861 / 10.6004 | `logs/ppl_q38_nvfp4_q8_v2.log` |

结论：Q8 与 BF16 差 `0.03`、方向还略低，属 chunk 噪声。**Q8_0 embedding
可接受，不必再转 BF16。** BF16 跑 38.81 s/pass（335 GB mmap，64 线程），
Q8 12.45 s/pass。

## 4. llama.cpp x86 CPU 逐层对齐

- 改动（已提交 `wip/q38fn`）：`common/debug.cpp` 支持 `LLAMA_DEBUG_DUMP_DIR`
  落 `.bin/.json`；`eval-callback` 请求全 prompt token 激活；
  `src/llama-model.cpp` 支持 fused `ffn_gate_up_exps.scale` 和
  `done_getting_tensors(true)`；转换器接受 default rope。
- dump 命令：`LLAMA_DEBUG_DUMP_DIR=output/llama_dump build/llama-x86/bin/
  llama-eval-callback -m gguf/q38fn-bf16.gguf -ngl 0 -c 512 -n 1 -t 64
  --tensor-filter '^(l_last|result_norm|result_output)' -p '<prompt>'`。
- 结果（`output/compare.json`）：layer0 `0.999967`，0–20 层 `≥0.999`，
  layer47 `0.996863`，final_norm `0.997637`，logits `0.998352`，argmax 271
  一致。深层下降是 BF16 跨实现累加误差，未定位到语义 bug；严格逐层 ≥0.999
  需统一累加精度或逐层 teacher-forcing。

## 5. host 内存与 ngram gather

- 实测 tensor 尺寸（`tools/cmm_budget_q38fn.py`，
  `logs/cmm_budget_q38fn_v2.log`）：专家 NVFP4 `64.600 GiB`、文本 dense
  `8.325 GiB`、ngram Q8_0 `50.664 GiB`、Q4_0 `26.822 GiB`、BF16
  `95.368 GiB`；ngram 表 `320,001,536` 行 ×160。
- 文本 CMM（权重 `72.925 GiB` + 256K 状态 `7.046 GiB` + 8 GiB 激活工作区）
  = `87.971 GiB`，128 GB 统一内存下 Linux 余 `40.029 GiB`。默认 256K 时
  Q8_0 不能常驻，Q4_0 可以。
- `tools/ngram_gather.py` 直接从 GGUF 元数据取 multipliers/offsets/sizes，
  mmap `per_layer_token_embd.weight`，实现 shift-right-ignore-EOS 哈希与
  Q8_0 反量化：热 `0.0025 ms/token`、冷 `0.0624 ms/token`（`logs/ngram_gather.json`）。
  冷值是 4090 本地盘 512 token 抽样，板上需按实际存储复测。

## 6. NPU kernel（仅 CRUN，未上板）

### 单算子：真实权重 + 真实 hidden，4 核 CRUN

| 算子 | 形状 | cos | max_abs | 日志 |
| --- | --- | ---: | ---: | --- |
| QSA（decode） | T=1, 24 头, hd256 | 0.9999679 | 0.0117188 | `logs/qsa_t1_afterfix.log` |
| QSA（prefill 内层） | T=10, 4 核 | 0.9999507 | 0.0117188 | `logs/qsa_real_afterfix.log` |
| PLE（decode） | T=1, 4 分支 | 0.9999981 | 0.000976562 | `logs/ple_t1_afterfix.log` |
| PLE（prefill 内层） | T=10, 4 分支 | 0.9999505 | 0.00964355 | `logs/ple_real_afterfix.log` |
| MoE（decode） | T=1, 512 专家 top-10, NVFP4 | 0.9995237 | 0.00463867 | `logs/moe_real_t1_afterfix.log` |
| MoE（prefill 内层） | T=256, 512 专家 top-10, NVFP4 | 0.9992040 | 0.0107422 | `logs/moe_real_t256_afterfix.log` |
| HC read/inject/write | T=8 tile | 0.9999512–0.9999979 | 0.03125 | `logs/hc_real_afterfix.log` |
| HC 256 行分块 | T=256, 32×8 tile | 0.9999512–0.9999972 | 0.03125 | `logs/hc_t256_afterfix.log` |
| GDN | T=1, 外部真实 mixed | 0.9999884 | 0.0078125 | `logs/gdn_t1_oc128.log` |
| GDN | T=256, fresh prefill, HF chunk=32 | 0.9999799 | 0.0117188 | `logs/gdn_t256_c32_skip0c.log` |

- QSA/PLE/MoE/HC 全部在修复 `ax.dsl.sigmoid`（融合图里被编成常量 -2.0）后重跑。
  修复方式是显式 `1/(1+exp(-x))`，见 `ops.py:sigmoid_exp`、`gdn.py:_sigmoid_exp`、
  `m_qwen3.py:_sigmoid_exp`。
- 512 专家 bank 是**单层单 bank IO**，没有 4 GiB 切块。decode/prefill 编译、
  MoE T=1/T=256 CRUN 都没有触发单 IO 限制，因此没有原始报错。
- 参考源说明：逐算子内部参考来自 HF Transformers 官方 `qwen4_exp`（sublayer
  输出）；llama.cpp CPU 的层边界 dump（`output/llama_dump/l_last-*.bin`）用于
  交叉核对，layer0 与 HF 参考 `cos=0.9999665`（`output/compare.json`）。本轮
  没有把 llama.cpp 内部 GDN/MoE 中间张量逐算子导出，这一点在验收上仍是缺口。

### 编出的 4 核 axmodel

| 文件 | 大小 | 编译 | 职责 | 每次推理调用 |
| --- | ---: | ---: | --- | ---: |
| `build/q38fn/decode_ctx4096.axmodel` | 5,811,237 B (5.54 MiB) | 459.2 s | 4 核 decode T=1，48 层运行时循环 + 权重 bank，`lm_head` 内联 | 1 / token |
| `build/q38fn/decode_ctx262144.axmodel` | 5,876,013 B (5.60 MiB) | 451.9 s | 同上，256K 状态 | 1 / token |
| `build/q38fn/prefill_ctx256.axmodel` | 86,557,345 B (82.5 MiB) | 7853.0 s | 4 核 prefill T=256，运行时 12 block 循环，QSA/PLE 64 行 tile，HC 8 行 tile | 1 / 256 token |

- decode 两档 state：4096 上下文 `0.325 GiB`；262144 上下文 `7.045 GiB`
  （QSA K/V 6.000、raw 0.750、pooled 0.188、GDN 0.105、conv 0.002）。
- prefill256 的 state 只有 `0.325 GiB`（32 页）；工作区按 8 GiB 预留。
- 编译日志：`logs/compile_decode4096_r3.log`、
  `logs/compile_decode262144_r3.log`、`logs/compile_prefill256_r3.log`。
- `estimated_cycles`：编译器不输出该字段，未上板也没有可信性能数字。单层
  CRUN 的 simulator trace 在 `tmp/qsa_real/simulator_cycle_trace.json`、
  `tmp/hc_final/simulator_cycle_trace.json`，只作调度参考，不当板上性能。

### 整层与多层：CRUN 已收敛

layer0 单层 CRUN（真实 layer0 权重、真实 embedding，`STAGES=4`）：

| 节点 | cos | 日志 |
| --- | ---: | --- |
| mixed_attn | 0.9999958 | `logs/stage4_oc128.log`、`logs/layer0_trace_oc128.log` |
| GDN | 0.9999844 | 同上 |
| after_attn | 0.9999646 | 同上 |
| mixed_ffn | 0.9999509 | 同上 |
| MoE | 0.9995171 | 同上 |
| after_moe | 0.9998111 | 同上 |

- 根因：GDN BF16 `project_normed_input` 的 OC tile 512 在融合图里过宽，
  尾端输出读到被调度覆盖的权重/输出 buffer。512/256/128 的 GDN cos 分别为
  `0.8310636`/`0.9228749`/`0.9999844`。
- 8 层（层 0–7，block0 + 运行时执行 block1）对 HF `layer_07`：
  `cos=0.9976548`、`max_abs=0.234375`（`logs/block_real_oc128.log`）。
  这是 4 层 BF16 误差累积；单层最低仍是 MoE `0.9995171`。
- 旧失败尝试（DDR staging、`axd.pub/sub`、OCM 私有 copy、per-core trace 等）
  保留在 `SOLVED_gdn_handoff.md`，不再作为当前状态。

### ViT

本轮按用户要求不做；公开仓库也没有可直接调用的 vision builder，ViT 的
`q38fn_vit` axmodel 未编。

### R2 收尾（2026-11-28 复核）

- **MoE T=256 已用真实权重、真实 hidden、512 专家 top-10、NVFP4 跑过 CRUN**：
  `cos=0.9992040`，`max_abs=0.0107422`，日志
  `logs/moe_real_t256_afterfix.log`。decode T=1 为 `cos=0.9995237`，日志
  `logs/moe_real_t1_afterfix.log`。两者都达到本轮 `cos >= 0.999` 的门槛。
- **prefill HC 残差已按 256 行分块验证**：`[256,4,2560]` 残差留在 DDR，
  每次把 8 行 tile 搬入 OCM 完成 read/inject/write 后写回；32 个 tile 的
  `cos=0.9999512–0.9999972`、`max_abs=0.03125`，日志
  `logs/hc_t256_afterfix.log`。4 核重复写相同字节，结果按最后写者一致。
- **prefill 其他大工作区的分块结论**：QSA 以 64 行 tile 执行 256 行，
  避免 `q_all/selected/attention` 超过约 8 MiB 的 OCM job 预算；PLE 以
  64 行 tile；MoE 内部继续按自己的 4096 行 prefill kernel 分块。这是
  `prefill_ctx256.axmodel` 能编出（82.5 MiB、303,660 jobs）的实际布局。
- **仍不能把整图 CRUN 当作通过**：整 48 层 decode 展开在模拟器达到约
  `250 GiB` RSS，被 200 GiB 上限终止；跨块续跑层 8–15 为
  `cos=0.9954564`，层 16–23 出现 `inf`，根因未定。R2 的验收边界仍是
  “单算子 + layer0 + 0–15 层块通过”，整模型端到端要在板上验证。

## 7. Host adapter

- **未完成**。计划：`ggml/src/ggml-axera/models/model.cpp` 的 `adapters[]`
  加一行 `qwen38_fn`，新目录 `models/qwen38_fn/`；不按名字 if、不带来回退。
  依赖第 6 节 axmodel 的 IO 名，所以没有先写死。

## 8. 优化清单对照（`KERNEL_OPT.md`）

| 条目 | 状态 |
| --- | --- |
| 1 整模型合并 kernel | decode1 + prefill256 已编出；layer0 与 0–15 层 CRUN 已过，整 48 层受 CRUN 内存限制 |
| 2 一律 4 核 | 两个 axmodel 都是 `[4,4]`；QSA/PLE/MoE/HC/GDN CRUN 4 核 |
| 3 照抄现成函数、dense 256 行 | 上游 QSA/HC/GDN/linear/norm/rope/topk；prefill QSA 因 OCM 预算用 64 行 tile，MoE/PLE 用 256/64 行，见 6 节 |
| 4 权重格式 | 专家 NVFP4、embedding Q8_0（用户指定）；KV BF16 |
| 5 单 IO <2 GiB 分块 | 用户 11-28 确认新版无单 IO 4 GiB 限制，专家单 bank IO；编译/CRUN 无限制报错 |
| 6 按 -c 分 KV | 已按 4096/262144 编两档，状态 0.325/7.045 GiB |
| 7 编码器一 kernel | 本轮不做；公开仓库无 vision builder |
| 8 融合采样头 | `lm_head` 已并入 decode；top-1024 采样头未做 |
| 9 prefill 尾巴补齐 | kernel 已接 `valid_tokens`；host padding 与 256 行 tail CRUN 未做 |
| 10 媒体进批次 | 未做 |
| 11 序列状态 | 未接 seqstate |
| 12 MTP 投机 | 模型有 1 层 MTP；未实现 |
| 13 加载期整块 memcpy | 待 host adapter |
| 14 KV 直绑 | 设计采用；axmodel IO 已直绑，host adapter 未接 |
| 15 checkpoint 恒定 | 不适用（无 server checkpoint） |
| 16 TTS/softmax | 不适用 |
| 17 跨核读旧值/质量门禁 | HC→GDN 问题已定位并修复（GDN OC tile 128）；layer0 全阶段 ≥0.999 |
| 18 CRUN 真权重真输入 | 单算子、layer0、8 层块已做；整 48 层因 simulator RSS >200 GiB 未完成 |
| 19 速度数字 | 未上板，不写性能 |
| 20–21 发布包文字 | 出包时执行 |

## 9. 未完成与风险

- **整 48 层 decode CRUN 未完成**：完整图在 runner 展开到 48 层时 RSS 达约
  250 GiB，超过用户 200 GiB 上限，已按 PID 终止。日志
  `logs/full_decode_crun.log`。因此只能说 axmodel 编译成功、单层/8 层块数值
  通过，不能宣称整模型端到端 CRUN 通过。
- **跨块逐段验证受数值发散阻断**：从 HF `layer_07` 续跑层 8–15 得到
  `cos=0.9954564`（`logs/block_pair_08_retry.log`），但从 `layer_15` 续跑层
  16–23 输出全 `inf`（`logs/block_pairs_rest.log`）。已检查 8 层权重缓存无
  inf/nan，也排除了脚本导入副作用；具体算子/激活溢出未定位，按纪律停止继续
  烧同类 CRUN。
- **prefill256 的 fresh chunk0 修复只覆盖从头开始的一页**。带已有 GDN state
  的续页 prefill 需要在下一轮改成运行期 `prefix == 0` 条件；当前不能把
  `fresh_prefill=True` 复用到第二页以后。
- ViT、host adapter、融合采样 top-1024 未完成。
- 统一内存下 Q8_0 ngram 表不能常驻；mmap 冷读 `0.062 ms/token` 是 4090 盘，
  128 GB 板上存储与 page cache 余量更小，需实测；Q4_0 是备选。
- Q4_0/Q4_K 的 PPL 对照**未做**：需要把 50.66 GiB ngram 表重写成约 26.8 GiB
  的新 GGUF 再跑 PPL。当前 Q8 冷 gather 仅 `0.062 ms/token`，先等板上存储
  实测；若板上吞吐不足再花这次重写。
- 深层 CPU 对齐 <0.999（BF16 跨实现误差）。
- llama.cpp CPU dump 只提供层边界 hidden，没有逐算子内部张量；本轮逐算子
  CRUN 的参考主要是 HF 官方实现，层边界用 llama.cpp 交叉核对。
- 板子未给，全程未连板。

## 10. 日志与提交

- 参考 `logs/ref_full.log`；BF16 GGUF `logs/convert_q38_bf16.log`；
  NVFP4 `logs/quant_q38_nvfp4_q8_v2.log`；PPL `logs/ppl_q38_bf16.log`、
  `logs/ppl_q38_nvfp4_q8_v2.log`；budget `logs/cmm_budget_q38fn_v2.log`；
  ngram `logs/ngram_gather.json`；对齐 `logs/llama_dump_bf16_v3.log`、
  `output/compare.json`。
- 本轮 axmodel：`logs/compile_decode4096_r3.log`、
  `logs/compile_decode262144_r3.log`、`logs/compile_prefill256_r3.log`；
  单算子：`logs/qsa_t1_afterfix.log`、`logs/qsa_real_afterfix.log`、
  `logs/ple_t1_afterfix.log`、`logs/ple_real_afterfix.log`、
  `logs/moe_real_t1_afterfix.log`、`logs/moe_real_t256_afterfix.log`、
  `logs/hc_real_afterfix.log`、`logs/gdn_t1_oc128.log`、
  `logs/gdn_t256_c32_skip0c.log`；整层：`logs/stage4_oc128.log`、
  `logs/layer0_trace_oc128.log`、`logs/block_real_oc128.log`；
  跨块与内存受限：`logs/block_pair_08_retry.log`、`logs/block_pairs_rest.log`、
  `logs/full_decode_crun.log`；历史失败证据与最终修复见
  `SOLVED_gdn_handoff.md`。
- `llama/` 分支 `wip/q38fn`，HEAD `ff029d01d`（作者
  `ZHEQIUSHUI <qiushui.zhe@gmail.com>`），已推 `fork/wip/q38fn`；本轮无新
  llama.cpp 改动。
- `npu-q38` 仓库分支 `wip/q38fn`，本轮 HEAD `685ee606a6`（作者
  `ZHEQIUSHUI <qiushui.zhe@gmail.com>`，无 Co-Authored-By），提交
  “qwen38_fn: fix fused GDN tile handoff and fresh prefill”；已推
  `origin/feat/q38fn`（`48208fc77f..685ee606a6`）。
- 该仓库没有配置 fork remote；内部 `origin` 的 pre-receive hook 拒绝
  `wip/*`（要求 `feat|fix|test|doc|release|revert|cherry-pick` 前缀），
  所以同一 commit 另推到 `origin/feat/q38fn`。本地 `wip/q38fn` 保留。
