# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import torch
import torch.nn.functional as F


def logmeanexp(t, dim):
    return torch.logsumexp(t, dim=dim) - torch.log(t.new_tensor(t.shape[dim]))


def masked_logmeanexp(t, mask, dim):
    expanded_mask = mask.unsqueeze(-1)
    masked_t = t.masked_fill(~expanded_mask, -torch.inf)
    count = mask.sum(dim=dim).clamp_min(1).to(t.dtype)
    return torch.logsumexp(masked_t, dim=dim) - count.log().unsqueeze(-1)


def masked_mean(t, mask, dim):
    expanded_mask = mask.unsqueeze(-1).to(t.dtype)
    count = expanded_mask.sum(dim=dim).clamp_min(1)
    return (t * expanded_mask).sum(dim=dim) / count


def masked_quantile(t, mask, q, default=0.0):
    values = t[mask]
    if values.numel() == 0:
        return t.new_tensor(default)
    return torch.quantile(values.float(), q)


def masked_rate(t, mask, default=0.0):
    if mask.sum().item() == 0:
        return t.new_tensor(default)
    return t[mask].float().mean()


def normalize_token_ids(token_ids, name, shape, device):
    n, m, x, _ = shape
    if token_ids is None:
        return None
    if torch.is_tensor(token_ids):
        token_ids = token_ids.to(device=device)
    else:
        token_ids = torch.tensor(token_ids, device=device)
    if tuple(token_ids.shape) == (n, x):
        token_ids = token_ids[:, None, :].expand(n, m, x)
    if tuple(token_ids.shape) != (n, m, x):
        raise ValueError(f"{name} must have shape [n, m, x] or [n, x]")
    return token_ids


def common_prefix_compare_mask(baseline_token_ids, candidate_token_ids):
    mismatch = baseline_token_ids != candidate_token_ids
    prev_mismatch_count = torch.cumsum(mismatch.long(), dim=-1) - mismatch.long()
    return prev_mismatch_count == 0


def js_div(log_p, log_q):
    # log_p/log_q: [..., h]
    p = log_p.exp()
    q = log_q.exp()
    m = 0.5 * (p + q)
    log_m = torch.log(m.clamp_min(1e-30))
    kl_pm = (p * (log_p - log_m)).sum(dim=-1)
    kl_qm = (q * (log_q - log_m)).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def guard_logits(
    baseline_logits,
    candidate_logits,
    *,
    baseline_token_ids=None,
    candidate_token_ids=None,
    alpha=3.0,
    gamma=2.0,
    abs_js_p99=1e-3,
    abs_js_p999=5e-3,
    large_margin=0.5,
    huge_margin=1.0,
    max_large_margin_flip_rate=0.0,
    max_huge_margin_top5_drop_rate=0.0,
    min_top10_overlap=0.8,
    max_centered_rel_l2_p99=0.05,
    max_first_divergence_rate=None,
    max_first_divergence_large_margin_rate=0.0,
    max_first_divergence_huge_margin_rate=0.0,
):
    """
    Return True if candidate is OK, False if abnormal.
    baseline_logits/candidate_logits: [n, m, x, h]

    If token ids are provided, logits after the first free-running token
    divergence are excluded. The divergence step itself is still checked.
    """

    assert baseline_logits.shape == candidate_logits.shape
    n, m, x, h = baseline_logits.shape
    if h < 2:
        raise ValueError("vocab dimension h must be at least 2")

    device = baseline_logits.device
    baseline_token_ids = normalize_token_ids(
        baseline_token_ids, "baseline_token_ids", baseline_logits.shape, device
    )
    candidate_token_ids = normalize_token_ids(
        candidate_token_ids, "candidate_token_ids", candidate_logits.shape, device
    )
    if (baseline_token_ids is None) != (candidate_token_ids is None):
        raise ValueError(
            "baseline_token_ids and candidate_token_ids must be provided together"
        )

    if baseline_token_ids is None:
        compare_mask = torch.ones((n, m, x), dtype=torch.bool, device=device)
        token_mismatch = torch.zeros((n, m, x), dtype=torch.bool, device=device)
    else:
        token_mismatch = baseline_token_ids != candidate_token_ids
        compare_mask = common_prefix_compare_mask(
            baseline_token_ids, candidate_token_ids
        )
    position_mask = compare_mask.any(dim=1)

    # 1. 用 log-prob 做主判断，规避 logits 整体平移。
    b_lp = F.log_softmax(baseline_logits.float(), dim=-1)
    c_lp = F.log_softmax(candidate_logits.float(), dim=-1)

    # 2. 每条样本、每个 token 的“重复运行平均分布”。
    # shape: [n, x, h]
    b_mean_lp = masked_logmeanexp(b_lp, compare_mask, dim=1)
    c_mean_lp = masked_logmeanexp(c_lp, compare_mask, dim=1)
    uniform_lp = b_mean_lp.new_full((1, 1, h), -math.log(h))
    b_mean_lp = torch.where(position_mask[..., None], b_mean_lp, uniform_lp)
    c_mean_lp = torch.where(position_mask[..., None], c_mean_lp, uniform_lp)

    # 3. baseline 自身噪声：每次 baseline run 到 baseline 平均分布的 JS。
    # shape: [n, m, x]
    b_noise = js_div(b_lp, b_mean_lp[:, None, :, :])

    # 4. candidate 自身噪声。
    c_noise = js_div(c_lp, c_mean_lp[:, None, :, :])

    # 5. candidate 相对 baseline 的中心漂移。
    # shape: [n, x]
    drift = js_div(b_mean_lp, c_mean_lp)

    b_noise_p99 = masked_quantile(b_noise, compare_mask, 0.99)
    b_noise_p999 = masked_quantile(b_noise, compare_mask, 0.999)

    c_noise_p99 = masked_quantile(c_noise, compare_mask, 0.99)
    drift_p99 = masked_quantile(drift, position_mask, 0.99)
    drift_p999 = masked_quantile(drift, position_mask, 0.999)

    # 6. large-margin top1 flip。
    topk = min(10, h)
    b_top = torch.topk(b_mean_lp, k=topk, dim=-1)
    c_top = torch.topk(c_mean_lp, k=topk, dim=-1)

    b_top1 = b_top.indices[..., 0]
    c_top1 = c_top.indices[..., 0]
    b_margin = b_top.values[..., 0] - b_top.values[..., 1]

    large_mask = b_margin > large_margin
    huge_mask = b_margin > huge_margin

    large_flip = (b_top1 != c_top1) & large_mask

    # huge margin 下，baseline top1 至少应该还在 candidate top5 内。
    c_top5 = c_top.indices[..., : min(5, topk)]
    b_top1_in_c_top5 = (c_top5 == b_top1[..., None]).any(dim=-1)
    huge_top5_drop = (~b_top1_in_c_top5) & huge_mask

    large_flip_rate = masked_rate(large_flip, position_mask)
    huge_top5_drop_rate = masked_rate(huge_top5_drop, position_mask)

    # 7. top10 overlap，防止概率分布尾部大面积乱掉。
    b_top10 = b_top.indices
    c_top10 = c_top.indices
    overlap = (
        (b_top10[..., :, None] == c_top10[..., None, :])
        .any(dim=-1)
        .float()
        .mean(dim=-1)
    )
    top10_overlap_mean = masked_rate(overlap, position_mask, default=1.0)

    # 8. centered logits diff，辅助捕捉 logits 形状/scale 的异常。
    b_center = baseline_logits.float() - baseline_logits.float().mean(
        dim=-1, keepdim=True
    )
    c_center = candidate_logits.float() - candidate_logits.float().mean(
        dim=-1, keepdim=True
    )

    b_center_mean = masked_mean(b_center, compare_mask, dim=1)
    c_center_mean = masked_mean(c_center, compare_mask, dim=1)

    centered_rel_l2 = torch.linalg.vector_norm(
        c_center_mean - b_center_mean, dim=-1
    ) / torch.linalg.vector_norm(b_center_mean, dim=-1).clamp_min(1e-6)
    centered_rel_l2_p99 = masked_quantile(centered_rel_l2, position_mask, 0.99)

    # 9. 自由生成时只用首次分叉判定行为异常。
    # 分叉后不做 logits diff。
    if baseline_token_ids is None:
        first_divergence_rate = baseline_logits.new_tensor(0.0)
        first_divergence_large_margin_rate = baseline_logits.new_tensor(0.0)
        first_divergence_huge_margin_rate = baseline_logits.new_tensor(0.0)
        mean_common_prefix_len = baseline_logits.new_tensor(float(x))
        mean_compared_token_fraction = baseline_logits.new_tensor(1.0)
    else:
        diverged = token_mismatch.any(dim=-1)
        first_divergence_idx = token_mismatch.long().argmax(dim=-1)
        common_prefix_len = torch.where(
            diverged, first_divergence_idx, torch.full_like(first_divergence_idx, x)
        )
        first_divergence_rate = diverged.float().mean()
        first_divergence_mask = token_mismatch & compare_mask

        b_run_top2 = torch.topk(b_lp, k=2, dim=-1)
        b_run_margin = b_run_top2.values[..., 0] - b_run_top2.values[..., 1]
        first_divergence_large_margin_rate = (
            first_divergence_mask & (b_run_margin > large_margin)
        ).any(dim=-1).float().mean()
        first_divergence_huge_margin_rate = (
            first_divergence_mask & (b_run_margin > huge_margin)
        ).any(dim=-1).float().mean()
        mean_common_prefix_len = common_prefix_len.float().mean()
        mean_compared_token_fraction = compare_mask.float().mean()

    # 10. 硬失败规则。
    checks = {
        "drift_p99_abs": drift_p99 <= abs_js_p99,
        "drift_p999_abs": drift_p999 <= abs_js_p999,
        "drift_vs_baseline_noise": (
            drift_p99 <= alpha * b_noise_p99.clamp_min(1e-12)
        ),
        "candidate_noise_vs_baseline": (
            c_noise_p99 <= gamma * b_noise_p99.clamp_min(1e-12)
        ),
        "large_margin_flip": large_flip_rate <= max_large_margin_flip_rate,
        "huge_margin_top5_drop": (
            huge_top5_drop_rate <= max_huge_margin_top5_drop_rate
        ),
        "top10_overlap": top10_overlap_mean >= min_top10_overlap,
        "centered_rel_l2": centered_rel_l2_p99 <= max_centered_rel_l2_p99,
        "first_divergence_large_margin": (
            first_divergence_large_margin_rate
            <= max_first_divergence_large_margin_rate
        ),
        "first_divergence_huge_margin": (
            first_divergence_huge_margin_rate
            <= max_first_divergence_huge_margin_rate
        ),
    }
    if max_first_divergence_rate is not None:
        checks["first_divergence_rate"] = (
            first_divergence_rate <= max_first_divergence_rate
        )

    ok = all(bool(v.item() if torch.is_tensor(v) else v) for v in checks.values())

    metrics = {
        "ok": ok,
        "mean_compared_token_fraction": float(mean_compared_token_fraction),
        "mean_common_prefix_len": float(mean_common_prefix_len),
        "b_noise_p99": float(b_noise_p99),
        "b_noise_p999": float(b_noise_p999),
        "c_noise_p99": float(c_noise_p99),
        "drift_p99": float(drift_p99),
        "drift_p999": float(drift_p999),
        "large_flip_rate": float(large_flip_rate),
        "huge_top5_drop_rate": float(huge_top5_drop_rate),
        "first_divergence_rate": float(first_divergence_rate),
        "first_divergence_large_margin_rate": float(
            first_divergence_large_margin_rate
        ),
        "first_divergence_huge_margin_rate": float(
            first_divergence_huge_margin_rate
        ),
        "top10_overlap_mean": float(top10_overlap_mean),
        "centered_rel_l2_p99": float(centered_rel_l2_p99),
        "failed_checks": [
            k
            for k, v in checks.items()
            if not bool(v.item() if torch.is_tensor(v) else v)
        ],
    }

    return ok, metrics
