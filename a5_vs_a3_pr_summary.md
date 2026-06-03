# A3 迁移到 A5 的整体思路与落地方案

本文用于梳理 `omni-npu` 从 A3 迁移到 A5/Ascend 950 的整体设计思路和具体落地方案。代码里的硬件分支主要通过 `on_ascend950()` 判断，因此下文将 `on_ascend950=True` 视为 A5 路径，将 `on_ascend950=False` 视为 A3/非 A5 的既有路径。

## 整体思路

A3 到 A5 的迁移不是简单复用原有算子和配置，而是围绕 Ascend 950 的算子接口、cache 写入方式和 attention metadata 重新组织关键路径。总体策略是：

- 保留 A3 既有逻辑，避免 A5 适配影响 A3 稳定性。
- 通过 `on_ascend950()` 将 A5 专用路径显式隔离。
- 对 attention、KV cache、MoME 等性能敏感模块采用 A5 专用算子接口。
- 对 sink attention 采用最终重构方案：A3 和 A5 逻辑拆成独立方法，A3 继续走 `_apply_sink_attention()`，A5 走 `_apply_sink_attention_a5()`。
- 对配置层新增 A5 best practice，让模型部署直接选择 A5 对应的 hybrid 和 1P1D 配置。

迁移后的目标是：A3 路径保持原有行为，A5 路径使用 Ascend 950 支持更好的 scatter、indexer、attention pioneer、fused causal conv 等接口，并让两套逻辑具备清晰的维护边界。

## 关键差异

| 模块 | A5 落地方式 | A3 保留方式 | 迁移目的 |
| --- | --- | --- | --- |
| 硬件分支 | 在 DSA、MLA 等模块引入 `on_ascend950()`，命中后进入 A5 专用逻辑。 | 不命中 `on_ascend950()` 时继续走原有逻辑。 | 将 A5 适配限定在明确分支内，降低对 A3 的回归风险。 |
| 推荐配置 | 新增 `openpangu_v2_35B`、`hardware=A5`、`precision=bf16` 配置入口，覆盖 hybrid 和 1P1D prefill/decode。 | A3 继续使用已有 best practice 配置。 | 让部署层能直接选择 A5 推荐参数。 |
| 配置参数 | A5 配置启用 `use_noncontiguous_kv`、`enable_kv_rmsnorm_rope_cache`，关闭 `kv_nz`、`merge_q_kv_conv`；hybrid 关闭 `enable_prefill_mla_absorb_pa`。 | A3 不复用这组 A5 配置。 | 对齐 A5 attention/cache 的算子能力和限制。 |
| DSA cache 更新 | A5 使用 `torch_npu.npu_scatter_nd_update_`，通过 `slot_mapping_2d` 更新 cache。 | A3 使用 `torch.ops.custom.npu_ai_infra_scatter_block_update_`。 | 适配 Ascend 950 支持的 scatter 写 cache 接口。 |
| DSA indexer | A5 non-contiguous KV 分支使用 `torch_npu.npu_lightning_indexer(**kwargs)[0]`。 | A3 使用 `torch.ops.custom.npu_lightning_indexer_enhance(...)`。 | 替换为 A5 兼容的 lightning indexer 算子路径。 |
| DSA/MLA KV cache | A5 禁用部分旧 fused KV RMSNorm RoPE cache 路径，改为手动 RMSNorm/RoPE 后用 `npu_scatter_nd_update_` 写入。 | A3 保留原 fused/custom cache 更新路径。 | 解决 A5 上旧 fused cache 算子不适配的问题。 |
| MLA sink attention | A5 使用 `_apply_sink_attention_a5()`，通过 `npu_ai_infra_attention_pioneer_metadata` 和 `npu_ai_infra_attention_pioneer` 执行，metadata 带 `soc_version="ascend950"`。 | A3 使用 `_apply_sink_attention()` 中的既有 sink/non-contiguous attention 逻辑。 | 将 A5 pioneer 协议和 A3 旧路径彻底拆开。 |
| MLA PA decode | A5 PA decode 使用 `input_layout="TND_NTD"`，传入 block table、query/key rope、sink KV/value/rope 等参数。 | A3 走原 PA/sink attention 参数组织。 | 对齐 A5 paged attention 的输入布局和 metadata 要求。 |
| Chunked prefill | A5 在 `enable_chunked_prefill && noncontiguous_kv && SWA` 场景下，从 paged cache 取历史 ctx KV，与当前 chunk 按 `[seq0_ctx, seq0_cur, seq1_ctx, seq1_cur...]` 交错拼接，再调用一次 attention。 | A3 不新增该 KV concat 路径。 | 用一次 FA 替代 LSE 拼接思路，提升 A5 chunked prefill 效率。 |
| MoME causal conv | A5 在 MoME RL 层中切换到 `torch_npu.npu_fused_causal_conv1d_v2`/`npu_fused_causal_conv1d`，并传入 block 调度 metadata。 | A3 继续使用 `torch.ops.custom.npu_ai_infra_fused_causal_conv1d`。 | 解决 A5 旧 MoME conv 接口可能卡死的问题。 |
| MoME 调用入口 | A5 不再在 patch 层区分 `forward_prefill`/`forward_decode`，统一调用 `conv_layer.forward(..., mome_metadata=metadata, inplace=False)`，在 layer 内部按硬件分支选择算子。 | A3 原本也是走 `forward()`。 | 让 MoME 调用入口收敛，硬件差异下沉到实现层。 |

## 具体落地方案

1. 配置层落地

   为 `openpangu_v2_35B` 新增 A5 BF16 best practice 配置，提供 hybrid、1P1D prefill、1P1D decode 三类配置文件。A5 配置重点启用 non-contiguous KV、KV RMSNorm RoPE cache 等能力，同时关闭 A5 不适合沿用的 `kv_nz`、`merge_q_kv_conv` 和 hybrid prefill MLA absorb PA。

2. Attention 路径落地

   在 DSA/MLA 中引入 A5 专用分支。A5 cache 写入使用 `slot_mapping_2d` 和 `npu_scatter_nd_update_`；DSA indexer 使用 `torch_npu.npu_lightning_indexer`；MLA sink attention 使用 attention pioneer metadata，显式传入 `soc_version="ascend950"`、`input_layout`、`actual_seq_lengths`、`actual_seq_lengths_kv`、sink KV/value/rope 等参数。

3. Sink attention 重构落地

   最终方案是将 A3/A5 sink attention 拆分为两个方法：

   - `_apply_sink_attention()`：保留 A3/非 A5 既有逻辑。
   - `_apply_sink_attention_a5()`：承载 A5 pioneer attention 逻辑，包括非 PA 和 PA 两种场景。

   入口处根据 `self.on_ascend950` 直接路由到 A5 方法。这样后续 A5 attention pioneer 参数演进不会和 A3 sink attention 逻辑互相牵连。

4. Chunked prefill 落地

   A5 chunked prefill 在 non-contiguous KV 且 SWA 的条件下，从 paged cache 中取历史上下文 KV，并与当前 chunk KV 交错拼接，重新生成 `kv_cumlens`。拼接顺序必须是按请求交错的 `[seq0_ctx, seq0_cur, seq1_ctx, seq1_cur...]`，避免将所有历史 KV 和所有当前 KV 分段拼在一起导致 attention 语义错误。

5. MoME 落地

   MoME causal conv 在 A5 上切换到新的 `torch_npu` fused causal conv 接口，patch 层统一调用 `forward()`，具体 prefill/decode 或 inplace/non-inplace 的差异由 layer 内部根据 metadata 和参数处理。

## 迁移注意事项

- A5 路径依赖 `slot_mapping_2d`、`block_table`、`seq_lens`、`kv_lens` 等 metadata；迁移时需要确认调度侧和 attention metadata 能完整提供这些字段。
- A5 chunked prefill KV concat 当前只支持 SWA 层；非 SWA 场景不应直接复用该路径。
- A5 sink attention 的 `soc_version="ascend950"`、`input_layout`、`actual_seq_lengths_kv`、sink KV/value/rope 参数必须与算子接口保持一致。
- A3 逻辑应尽量保持在原方法内，不把 A5 pioneer 分支混回 A3 方法，避免后续维护时互相影响。
- MoME 新接口需要重点关注 prefill/decode、inplace/non-inplace、block index、pad slot、num accepted/computed tokens 等 metadata 是否和算子期望一致。

## 相关 PR

这些 PR 是迁移方案的代码来源和阶段性实现参考：

- https://gitee.com/omniai/omni-npu/pulls/1158
- https://gitee.com/omniai/omni-npu/pulls/1195
- https://gitee.com/omniai/omni-npu/pulls/1372
- https://gitee.com/omniai/omni-npu/pulls/1375
