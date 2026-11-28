# 已解决：单图内 HC read → GDN 的数值不一致

## 结论

上一轮记录的 `HC read -> GDN cos=0.8310636` 已修复。根因不是 HC/GDN
数学，也不是 tensor handoff 本身；是 **BF16 GDN 投影的 OC tile 过大**：
`project_normed_input` 的 BF16 分支使用 `oc_tile_width=512` 时，融合图里
尾端输出会读到被 later op 覆盖的 runtime-bank buffer。改为 **128** 后，
同一张 layer0 图的 GDN 从 `0.8310636` 升到 `0.9999844`。

同轮还关闭了 fresh prefill 的首个 32-token chunk 问题：首 chunk 的 recurrent
state 恒为零，但零 BF16 staging tensor 在融合图中的生命周期不可靠；现在
`fresh_prefill=True` 且 `chunk_start == 0` 时跳过 state product，只计算
`k^T v` 路径。GDN T=256 全 chunk 达到 `0.9999799`。

## 最终证据（真实权重、真实 hidden、4 核 CRUN）

| 验证 | cos | 日志 |
| --- | ---: | --- |
| layer0 mixed_attn | 0.9999958 | `logs/stage4_oc128.log` |
| layer0 GDN | 0.9999844 | 同上 |
| layer0 after_attn | 0.9999646 | 同上 |
| layer0 mixed_ffn | 0.9999509 | 同上 |
| layer0 MoE | 0.9995171 | 同上 |
| layer0 after_moe | 0.9998111 | 同上 |
| GDN T=1 | 0.9999884 | `logs/gdn_t1_oc128.log` |
| GDN T=256（HF chunk=32） | 0.9999799 | `logs/gdn_t256_c32_skip0c.log` |
| 8 层（block0 + 运行时执行 block1） | 0.9976548 | `logs/block_real_oc128.log` |

8 层 block 的 `0.9976548` 是 BF16 跨层累积误差，不是单层功能失败；单层最低
MoE 仍为 `0.9995171`。

## 旧失败证据（保留）

- `oc_tile_width=512`：`logs/bisect_internal.log`，GDN `0.8310636`。
- `oc_tile_width=256`：`logs/bisect_oc256.log`，GDN `0.9228749`。
- `oc_tile_width=128`：`logs/stage4_oc128.log`，GDN `0.9999844`。
- 其它无效尝试：DDR staging、`axd.pub/sub`、OCM 私有 copy、fresh banks、
  per-core trace、`sync_all`；都不是最终修复。

## 当前限制

- 预编译的 prefill256 走 fresh prefill 修复；**从第二页开始带已有 GDN state
  的续写 prefill 尚未做独立验收**。它不能直接复用 `fresh_prefill=True` 的
  chunk0 跳过路径，下一轮需要改成运行期 `prefix == 0` 条件或等价安全路径。
- 板子未给，以上全部是 CRUN，不是板上 PyRun/实机结论。
