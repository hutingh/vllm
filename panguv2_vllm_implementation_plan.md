# PanguV2 vLLM 实现方案

本文档整理 `/data/public/huting/vllm` 中从
`39910f2b25aacc09f5e7f166cdf0030b19f8b9e8` 之后围绕 PanguV2 的主要适配。
汇报时建议先由我做整体方案开场，然后重点介绍 MoME layer 与
`causal_conv1d` kernel 的适配，之后切到各模块负责人继续展开。

## 1. 汇报分工与讲解顺序

整体方案分为以下几块：

- 开场总结与 MoME layer 详细方案：我负责
- KV cache manager / KV cache spec：毛润泽
- Attention backend：袁淘
- MTP / spec decode：袁淘
- mHC：于昌辉
- tool parser：于昌辉

建议讲解顺序：

1. 我先讲 PanguV2 vLLM 适配的总体目标和整体执行链路。
2. 我重点讲 MoME layer，包括它在 MLA 中的位置、state cache 设计、metadata builder，以及如何与 `causal_conv1d_fn` / `causal_conv1d_update` 结合。
3. MoME 部分讲完后，切到 KV cache manager，由毛润泽继续讲 cache spec、block manager 和 raw cache reshape。
4. 再切到 attention backend 和 MTP，由袁淘讲 static sink MLA、sparse MLA、spec decode 和 multi KV cache group proposer。
5. 最后由于昌辉讲 mHC 和 tool parser 相关适配。

## 2. 总体方案概览

PanguV2 的 vLLM 适配目标是让 OpenPanguV2 在 vLLM v1 执行路径中支持：

- MLA / sparse MLA / DSA attention
- static sink attention
- MoME short convolution
- mHC
- MTP speculative decoding
- tool parser 相关推理入口适配

这次改动不是单纯新增一个模型文件，而是把 PanguV2 的特殊结构接入 vLLM v1 的标准 runtime：

- 用新的 `KVCacheSpec` 描述 MLA、DSA、Sink、MoME state 的内存形态。
- 用 vLLM KV cache manager 统一管理 block 生命周期和 sliding-window state。
- 用 MoME 自己的 metadata builder 连接 continuous batching、chunked prefill、decode、CUDA graph 和 spec decode。
- 用 attention backend override 支持 static sink 和 sparse/dense MLA 组合。
- 用 multi KV cache group proposer 支持 MTP draft model 中不同 cache group 的独立 block table 和 slot mapping。

核心执行链路可以概括为：

```text
OpenPanguDecoderLayer
  └── OpenPanguMLAAttention
        ├── fused_qkv_a_proj / q_a_layernorm / q_b_proj
        ├── mome_attn(q_c, state_indice=0)
        ├── kv_a_layernorm
        ├── mome_attn(kv_c, state_indice=1)
        ├── StaticSinkMLAAttention / sparse MLA backend
        ├── mome_attn(attn_out, state_indice=2)
        └── o_proj
```

其中 MoME 是我主要负责讲的部分。它虽然在模型结构上表现为 attention block 内部的短卷积残差分支，但在 vLLM runtime 中需要像 Mamba state 一样参与 KV cache 分配、block table 映射、prefill/decode 状态更新和 spec decode accepted token 更新。

## 3. MoME Layer 适配方案

### 3.1 MoME 在模型结构中的位置

MoME 不是一个独立 decoder block，而是 MLA attention 内部的三处短卷积残差增强。它分别作用在 q lora、compressed kv 和 attention output 三个位置。

q lora 分支：

```python
q_c = mome_attn(q_c, state_indice=0) + q_c
q_c = q_a_layernorm(q_c)
q = q_b_proj(q_c)
```

compressed kv 分支：

```python
kv_c = mome_attn(kv_c, state_indice=1) + kv_c
kv_c_normed = kv_a_layernorm(kv_c)
```

attention output 分支：

```python
attn_out = mla_attn(...)
attn_out = mome_attn(attn_out, state_indice=2) + attn_out
output = o_proj(attn_out)
```

这三个位置共享同一个 `MomeAttention` 模块，但通过 `state_indice` 选择不同的 conv weight 和不同的 conv state。这样既能复用同一套 metadata，也能保证三段 state 在 cache 中独立存储。

MoME 的三个 short-conv state 分别对应：

- `state_indice=0`: q lora 分支，维度为 `q_lora_rank`
- `state_indice=1`: compressed kv 分支，维度为 `kv_lora_rank`
- `state_indice=2`: attention output 分支，维度为 `num_local_heads * v_head_dim`

### 3.2 MomeAttention 模块设计

`MomeAttention` 被实现为 `AttentionLayerBase + CustomOp`。这点很重要，因为它不是普通的 `nn.Module`：它需要被 vLLM 识别为一种拥有 KV cache spec、attention backend 和 metadata builder 的状态层。

模块内部有三个 depthwise conv：

```python
self.qa_conv = nn.Conv1d(q_lora_rank, q_lora_rank, kernel_size,
                         groups=q_lora_rank, bias=False)
self.compresskv_conv = nn.Conv1d(kv_lora_rank, kv_lora_rank, kernel_size,
                                 groups=kv_lora_rank, bias=False)
self.o_conv = nn.Conv1d(o_dim, o_dim, kernel_size,
                        groups=o_dim, bias=False)
```

三个 conv 分别服务 q、compressed kv、output 三个位置。它们都是 channel-wise short convolution，所以权重在执行前 reshape 为：

```python
conv_weight = conv_weight.view(hidden_size, kernel_size)
```

`MomeAttention.forward` 的核心逻辑是：

```python
if state_indice == 0:
    conv_weight = qa_conv.weight
    hidden_size = q_lora_rank
elif state_indice == 1:
    conv_weight = compresskv_conv.weight
    hidden_size = kv_lora_rank
else:
    conv_weight = o_conv.weight
    hidden_size = o_dim

conv_state = self.kv_cache[state_indice]
torch.ops.vllm.mome_attention_fused_op(
    hidden_states,
    conv_state,
    conv_weight,
    hidden_size,
    self.prefix,
    output,
)
```

这里 `conv_state` 不由模块自己创建，而是由 vLLM KV cache manager 在运行时注入。因此 MoME state 的生命周期和 request/block 生命周期保持一致，不需要模型层维护额外的 request-state 字典。

### 3.3 MoME state cache 设计

MoME 的 KV cache spec 是 `SlidingWindowMomeSpec`。它继承 sliding-window 相关语义，但真正存储的不是 attention KV，而是三段 short-conv state：

```python
SlidingWindowMomeSpec(
    block_size=8,
    num_kv_heads=1,
    head_size=q_lora_rank + kv_lora_rank + o_dim,
    sliding_window=8,
    component_dims=(q_lora_rank, kv_lora_rank, o_dim),
)
```

逻辑上可以理解为：

```text
raw cache page
  ├── q conv state
  ├── compressed kv conv state
  └── output conv state
```

每个 state tensor 的形状是：

```text
(num_blocks, block_size, component_dim)
```

当前实现中 `block_size=8` 和 `sliding_window=8` 仍是 hardcode，后续可以从 PanguV2 config 中读取。这里先固定为 8，是为了和 MoME short-conv state 更新粒度以及 block aligned update 保持一致。

### 3.4 causal_conv1d_fn / causal_conv1d_update 与 MoME cache 使用模式

MoME 本质上是 depthwise short convolution。它需要维护每个 request 的短卷积历史状态，并在每次 prefill 或 decode 时根据新 token 更新 state。vLLM 里已有 Mamba/short-conv 相关的 causal conv kernel，因此我们的设计不是重新写一套 MoME kernel，而是复用两类 kernel：

- `causal_conv1d_fn`：用于 prefill / chunked prefill 的 varlen forward。输入是一段连续 token，kernel 一边计算 short-conv 输出，一边把每个 sequence 末尾的 conv state 写回 cache。
- `causal_conv1d_update`：用于 decode / spec decode 的 incremental update。输入通常是每个 request 的一个或多个新 token，kernel 从已有 state 读历史，再把 accepted token 推进到 persistent state。

两者的分工可以概括为：

```text
prefill / chunked prefill
  hidden_states[seq tokens]
    -> causal_conv1d_fn
       - 按 query_start_loc 切分 varlen sequences
       - 按 BLOCK_M 切分 sequence chunk
       - 计算 MoME residual
       - 写回 chunk 末尾 conv state

decode / spec decode
  hidden_states[new tokens]
    -> causal_conv1d_update
       - 从 block table 找到当前 request 的 state block
       - 基于 last computed state 做增量卷积
       - 只把真实推进的 token 写回 conv state
```

MoME + sliding window 可以复用这两个 kernel，关键原因是 MoME state 的生命周期和 sliding-window block 生命周期是兼容的：

- MoME 只需要最近 `kernel_size - 1` 个 token 的 short-conv state。
- sliding-window cache 本来就按 request 的最近 token/block 维护状态。
- vLLM 的 block table 已经能表达 request 到 cache block 的映射。
- causal conv kernel 已经支持通过 `cache_indices` / `conv_state_indices` 从 cache 中读取和写回 state。

因此我们把 MoME 的三段 state 也做成 sliding-window KV cache spec，而不是在模型层维护私有状态。

MoME cache 的使用模式与 Mamba 的区别主要在 block 访问方式上。Mamba 非 APC 模式可以把 state 视为一个 request 反复更新同一个 cache block；而 MoME 走 sliding-window cache 后，访问位置会随着 token 推进持续向后偏移，窗口前面的 block 会被 sliding window manager 释放。这一点更接近 Mamba 的 APC 模式：state 是按 block 序列推进的，而不是固定写一个 block。

```text
Mamba 非 APC 模式：固定 state block，反复原地更新

request A
  step t0        step t1        step t2        step t3
     │             │             │             │
     ▼             ▼             ▼             ▼
  ┌────────────────────────────────────────────────┐
  │                state block #0                  │
  │        read old state -> write new state        │
  └────────────────────────────────────────────────┘

特点：
  - 一个 request 主要复用同一个 state cache line/block。
  - cache 位置不随 token 位置向后移动。
  - 不需要 sliding-window block 释放语义。
```

```text
Mamba APC 模式 / MoME sliding-window 模式：访问随 token/block 向后推进

sequence tokens:
  [ block 0 ][ block 1 ][ block 2 ][ block 3 ][ block 4 ] ...
       │         │         │         │         │
       ▼         ▼         ▼         ▼         ▼
cache blocks:
  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
  │ state 0 │ │ state 1 │ │ state 2 │ │ state 3 │ │ state 4 │
  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
       ▲         ▲         ▲
       │         │         │
  sliding-window active range, window moves right

when window moves:
  [ block 0 ] is released
              [ block 1 ][ block 2 ][ block 3 ] become active range

特点：
  - block table 表达 request 当前可访问的 state block 序列。
  - prefill/decode 都根据 block index 找到 last computed / scheduled block。
  - 窗口向后滑动时，窗口前的 block 可以释放并复用。
  - 这与 MoME 的 short-conv state 按时间推进的语义一致。
```

MoME 在每个 sliding-window block 内又拆出三段 state：

```text
SlidingWindowMomeSpec cache group
        │
        ▼
per block raw cache page
        │
        ├── q conv state            state_indice=0
        ├── compressed-kv conv state state_indice=1
        └── o conv state            state_indice=2

block table / block indices
        │
        ├── block_idx_last_computed_token: 读取历史 state
        ├── block_idx_first_scheduled_token: prefill 开始写入位置
        └── block_idx_last_scheduled_token: 本轮最后写入位置
        │
        ▼
mome_attention_fused_op
        ├── prefill: causal_conv1d_fn
        └── decode/spec decode: causal_conv1d_update
```

因此这里复用 causal conv kernel 的关键不是“固定 state block”，而是使用它已经支持的 APC/block-index 参数，把 MoME 的 sliding-window block table 传进去，让 kernel 按向后推进的 block 序列读写 state。

KV cache reshape 后，MoME state 的逻辑 layout 是：

```text
(num_blocks, block_size, hidden_size)
```

而 causal conv kernel 使用的 conv state layout 是：

```text
(num_blocks, hidden_size, state_len)
```

因此 fused op 中会先做 layout 转换：

```python
conv_state_ = conv_state.transpose(-1, -2)
```

之后根据 metadata 判断当前 batch 中有多少 decode token 和 prefill token。decode 部分走 `causal_conv1d_update`，prefill 部分走 `causal_conv1d_fn`。

prefill 调用大致是：

```python
causal_conv1d_fn(
    prefill_hidden_states.transpose(0, 1),
    conv_weight,
    conv_states=conv_state_,
    has_initial_state=has_initial_states_p,
    cache_indices=state_indices_tensor_p,
    query_start_loc=query_start_loc_p,
    block_idx_first_scheduled_token=block_idx_first_scheduled_token_p,
    block_idx_last_scheduled_token=block_idx_last_scheduled_token_p,
    initial_state_idx=block_idx_last_computed_token_p,
    num_computed_tokens=num_computed_tokens_p,
    block_size_to_align=conv_state_.size(-1),
    metadata=mome_metadata,
    zero_initial_state_output=True,
)
```

decode 调用大致是：

```python
causal_conv1d_update(
    decode_hidden_states,
    conv_state_,
    conv_weight,
    conv_state_indices=state_indices_tensor_d,
    block_idx_last_scheduled_token=block_idx_last_scheduled_token_d,
    initial_state_idx=block_idx_last_computed_token_d,
    num_accepted_tokens=num_accepted_tokens,
    query_start_loc=query_start_loc_d,
    max_query_len=max_decode_query_len,
)
```

这里 spec decode 下的 `num_accepted_tokens` 很关键。MoME state 是递推状态，如果 rejected draft token 被写入 state，后续 token 都会被污染。因此 decode kernel 只能根据 accepted token 数量推进 persistent state。

我们对 `causal_conv1d_fn` 也做了一处 MoME 必需的修改：新增 `zero_initial_state_output` 参数，并传给 Triton kernel：

```python
ZERO_INITIAL_STATE_OUTPUT=zero_initial_state_output
```

kernel 内部新增逻辑：

```python
if ZERO_INITIAL_STATE_OUTPUT and chunk_offset == 0:
    zero_warmup = (load_init_state == 0) & (idx_token < state_len)
    acc = tl.where(zero_warmup, 0.0, acc)
```

这个修改用于处理没有历史 state 的新 request。short-conv 的前 `state_len` 个 token 属于 warmup 区间，如果直接按零 initial state 卷积，会得到一段由当前输入和空 state 混合产生的输出；但 MoME residual 的预期语义是没有足够历史 token 时不引入有效增量。因此 warmup 输出置 0，外层再做：

```python
hidden = mome_output + hidden
```

这样 warmup token 等价于保留原始 hidden states，不会被空 state 偏移。

另外 FULL CUDA graph 会对 batch token 做 padding。为了避免 padded tail 的 output 未初始化，`mome_attention_fused_op` 会先执行：

```python
output.copy_(hidden_states)
```

然后只覆盖真实 token 范围。这样 padded token 保持 identity 输出，不会把随机值传给后续 layer 或 collective。

### 3.5 MoME backend / metadata builder 如何落地 3.3 的 cache 方案

为了让 3.3 中的 `SlidingWindowMomeSpec` 真正工作，需要把 MoME 接入 vLLM 的 attention backend 和 metadata builder 体系。实现上新增 `MomeAttentionBackend` 和 `MomeAttentionMetadataBuilder`。

`MomeAttentionBackend` 的职责很轻，主要是把 MoME layer 关联到自己的 builder：

```python
class MomeAttentionBackend(AttentionBackend):
    @staticmethod
    def get_builder_cls():
        return MomeAttentionMetadataBuilder
```

真正的核心在 `MomeAttentionMetadataBuilder`。它不直接继承 Mamba builder，原因是 MoME 的 cache 使用模式和 Mamba 不完全一样：

- MoME 使用 `SlidingWindowManager` 管 cache ownership。
- causal conv kernel 需要显式 read/write block offset。
- Mamba 的部分 cache mode 会折叠 block table，不适合 MoME 的 block-level state。
- spec decode 下需要把 accepted token 数量传到 kernel，保证 rejected token 不写入 state。

builder 的构造过程可以理解为五步：

```text
CommonAttentionMetadata
  │
  ├─ 1. single-token prefill with prior state -> decode
  │
  ├─ 2. split_decodes_and_prefills
  │
  ├─ 3. compute block indices
  │      - block_idx_last_computed_token
  │      - block_idx_first_scheduled_token
  │      - block_idx_last_scheduled_token
  │
  ├─ 4. build prefill/decode metadata tensors
  │      - state_indices_tensor_p / d
  │      - query_start_loc_p / d
  │      - num_computed_tokens_p
  │      - num_accepted_tokens
  │
  └─ 5. full CUDA graph padding metadata update
```

metadata 中保存 decode 和 prefill 两套信息：

```text
decode:
  state_indices_tensor_d
  block_idx_last_scheduled_token_d
  block_idx_last_computed_token_d
  query_start_loc_d
  num_accepted_tokens
  max_decode_query_len

prefill:
  has_initial_states_p
  query_start_loc_p
  num_computed_tokens_p
  state_indices_tensor_p
  block_idx_first_scheduled_token_p
  block_idx_last_scheduled_token_p
  block_idx_last_computed_token_p
  nums_dict / batch_ptr / token_chunk_offset_ptr
```

block index 的计算方式是：

```python
block_idx_last_computed_token = cdiv(num_computed_tokens, block_size) - 1
block_idx_first_scheduled_token = cdiv(num_computed_tokens + 1, block_size) - 1
block_idx_last_scheduled_token = cdiv(seq_lens, block_size) - 1
```

这些 index 对应了 kernel 要读写的 state block：

- `block_idx_last_computed_token`：已有历史 state 所在 block。
- `block_idx_first_scheduled_token`：本轮 prefill 需要开始写入的 block。
- `block_idx_last_scheduled_token`：本轮 prefill/decode 需要写到的最后 block。

prefill 还需要调用：

```python
compute_causal_conv1d_metadata(query_start_loc_p_cpu, device=...)
```

生成 `nums_dict`、`batch_ptr`、`token_chunk_offset_ptr`。这些结构用于告诉 Triton kernel 每个 program 处理哪个 sequence、哪个 chunk，以及 chunk 在 sequence 内的 token offset。

builder 还有一个特殊处理：

```python
_treat_single_token_prefills_as_decodes(...)
```

当 request 被标记为 prefill，但 query length 为 1 且已有 prior state 时，它对 MoME 来说实际是 decode。将它改成 decode 后，可以让它走 `causal_conv1d_update`，避免 prefill kernel 按新 chunk 处理而导致状态错位。

最后，`MomeAttention.forward` 通过 custom op 进入 fused op：

```python
torch.ops.vllm.mome_attention_fused_op(...)
```

fused op 通过 `layer_name` 从 forward context 中取到 builder 生成的 metadata，再选择 prefill/decode kernel。这就是 3.3 的 cache spec、vLLM block table、MoME layer 和 causal conv kernel 串起来的完整路径。

## 4. 后续模块交接内容

下面几节作为我讲完 MoME 后的交接内容，后续由对应负责人展开。

## 5. KV Cache Manager / KV Cache Spec

负责人：毛润泽。

PanguV2 需要多种非标准 KV cache 形态，主要新增或扩展了：

- `SlidingWindowMomeSpec`
- `DSAAttentionSpec`
- `SinkDSAAttentionSpec`
- `SinkMLAAttentionSpec`
- `MLASlidingWindowSpec`
- `SinkMLASlidingWindowSpec`

`single_type_kv_cache_manager.py` 中将新增 spec 接入 manager：

- `SlidingWindowMomeSpec -> SlidingWindowManager`
- `DSAAttentionSpec -> FullAttentionManager`
- `SinkMLAAttentionSpec -> SinkFullAttentionManager`
- `SinkMLASlidingWindowSpec -> SinkSlidingWindowManager`
- `SinkDSAAttentionSpec -> SinkFullAttentionManager`

`GPUModelRunner._reshape_kv_cache_tensors` 中针对 `SlidingWindowMomeSpec` 做特殊 reshape。普通 attention KV cache 会 reshape 成 backend 需要的 KV layout；MoME 则按照 `component_dims` 将同一块 raw tensor 切成三个 state tensor。

另外 `bind_kv_cache` 支持 composite cache：如果 cache 是 `(mla_cache, indexer_cache)`，则分别绑定给 attention layer 和 indexer。

## 6. Attention Backend

负责人：袁淘。

attention backend 主要解决 PanguV2 的 static sink MLA、sparse MLA/DSA、sliding window、chunked prefill 等组合。

主要改动：

- 新增 `StaticSinkMLAAttention`，基于普通 `MLAAttention`，但通过 `create_static_sink_attention_backend` 挂接 sink-aware impl。
- `create_static_sink_attention_backend` 改为 backend override：override `reshape_kv_cache`，并根据底层 impl 替换为 sink impl。
- 新增或扩展 `FlashAttnStaticSinkMLAImpl`。
- 新增或扩展 `FlashMLASparseStaticSinkImpl`。
- attention backend registry 新增 `MOME`。

static sink 的处理不再通过修改通用 block table 来伪造 prefix token，而是在 MLA backend 内部显式计算：

```text
sink attention output + normal/sparse/sliding-window attention output
  -> merge_attn_states(LSE merge)
```

这样可以避免污染通用 metadata，也更容易适配 chunked prefill、sliding window、sparse MLA 和 decode。

## 7. MTP / Spec Decode

负责人：袁淘。

MTP 适配主要是让 OpenPanguV2 MTP 作为 draft model 工作，并支持 draft layers 跨多个 KV cache group。

主要改动：

- `SpeculativeConfig` 新增 `openpangu_mtp`。
- `openpangu_v2` 会转换为 `openpangu_mtp`。
- architecture 设置为 `OpenPanguMTPModel`。
- `n_predict` 来自 `num_nextn_predict_layers`。
- 新增 `OpenpanguMTPModelArchConfigConvertor`，让 MTP 模型 hidden layers 数量取 `num_nextn_predict_layers`。
- `OpenPanguMTP.load_weights` 适配 MoE 和 MoME 权重命名。
- `SpecDecodeBaseProposer` 移除“所有 draft layers 必须属于同一个 KV cache group”的假设，改为 per-group block table、slot mapping 和 metadata builder。

OpenPanguV2 draft model 中，draft layers 可能横跨多个 KV cache group：

- MLA attention KV cache group
- MoME state cache group
- 可能存在的 sink/sparse/indexer cache group

`GPUModelRunner` 在准备输入时会调用：

```python
drafter.set_per_group_block_table(kv_cache_gid, block_table)
drafter.set_per_group_slot_mapping(kv_cache_gid, slot_mapping)
drafter.set_draft_num_accepted_tokens(num_accepted_tokens)
```

这使得 MoME metadata builder 能拿到与自己 KV cache group 对应的 block table，而不是误用 attention KV 的 block table。spec decode 下，accepted token 数量也会传入 `causal_conv1d_update`，保证 rejected draft token 不会写入 MoME state。

## 8. mHC

负责人：于昌辉。

mHC 相关适配集中在 OpenPangu 模型结构中，主要包括：

- 新增 `mHCModule` custom op。
- decoder layer 中区分 normal forward 和 `forward_mhc`。
- attention 前后、MLP 前后分别通过 mHC pre/post op 做 stream mixing。
- 非 MTP layer 使用 mHC，MTP layer 跳过对应逻辑。
- 最终 head 前通过 merge mHC module 和 `HCHeadOp` 合并 stream 表示。

这一部分建议单独介绍 mHC 的数据流：

```text
hidden_states
  -> attn_mhc_module.hc_pre
  -> input_layernorm
  -> self_attn
  -> attn_mhc_module.hc_post
  -> mlp_mhc_module.hc_pre
  -> pre_mlp_layernorm
  -> mlp
  -> mlp_mhc_module.hc_post
```

## 9. Tool Parser

负责人：于昌辉。

tool parser 属于 PanguV2 推理入口和输出解析相关适配。建议在汇报中独立说明：

- PanguV2 tool call 输出格式。
- vLLM OpenAI server / chat completion 路径如何识别 tool call。
- parser 如何从模型输出中抽取 tool name、arguments 和普通文本。
- 与 tokenizer、chat template、streaming output 的兼容点。

如果后续文档要补代码级细节，可以把 tool parser 的具体文件、解析状态机和异常 fallback 单独展开。

## 10. 当前限制与后续优化

当前实现里仍有一些可以后续清理或增强的点：

- `MomeAttention.get_kv_cache_spec()` 中 `block_size=8` 和 `sliding_window=8` 仍是 hardcode，建议后续从 PanguV2 config 中读取。
- MoME 相关实现目前和 causal_conv1d 的 layout、block 对齐强绑定，需要在方案中明确 kernel width、state len、block size 的约束。
- sparse static sink 的 fp8 路径仍有限制，需要单独评估 `fp8_ds_mla` 下 composite cache layout。
- 当前工作区中存在调试 print，合入前建议清理，避免影响性能和日志可读性。
- multi KV cache group spec decode 已打通核心路径，但仍建议补充覆盖：pure decode、chunked prefill、full CUDA graph decode、MTP accepted/rejected token 混合场景、MoME state warmup 场景。

## 11. 总结

PanguV2 vLLM 适配的核心是把 PanguV2 的特殊结构接入 vLLM v1 的标准 runtime。我这部分重点讲 MoME，因为它同时跨越模型结构、KV cache、metadata builder 和 kernel state update。MoME 的关键设计是：把 short-conv state 当作 vLLM KV cache 管理的状态页，而不是模型层私有状态。这样它才能在 continuous batching、chunked prefill、CUDA graph 和 spec decode 下保持一致的 request/block 生命周期。

MoME 讲完后，后续 KV cache manager、attention backend、MTP、mHC 和 tool parser 可以按负责人继续展开。
