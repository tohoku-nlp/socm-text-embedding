import argparse
import math
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from utils.mylogging import init_logger
from utils.seed import fix_seeds

logger = init_logger(__name__, log_file_path=Path("logs/calc_lambda.log"))
fix_seeds(0)


# In row-major code, context = A @ V, so centering over tokens is P @ A.
# In column-major theory Y = W^o W^v H A^T, the corresponding factor is A^T @ P.
# Since P is symmetric, ||P A||_F = ||A^T P||_F.
def compute_pa_fro_sq(A: np.ndarray) -> float:
    """
    Compute ||PA||_F^2 where P = I - (1/n)11^T.

    Parameters
    ----------
    A : np.ndarray
        Square matrix of shape (n, n).

    Returns
    -------
    float
        ||PA||_F^2.
    """
    A = A.astype(np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D matrix.")

    n = A.shape[0]
    one = np.ones((n, 1), dtype=np.float64)
    P = np.eye(n, dtype=np.float64) - (one @ one.T) / n
    PA = P @ A

    return float(np.sum(PA**2))


def get_output_projection(layer) -> torch.Tensor:
    """
    Extract per-head output projection weights.

    Parameters
    ----------
    layer : BertLayer
        A single BERT encoder layer.

    Returns
    -------
    torch.Tensor
        Wo_heads of shape (num_heads, hidden_dim, head_dim).
    """
    self_attn = layer.attention.self
    num_heads = self_attn.num_attention_heads
    head_dim = self_attn.attention_head_size

    Wo = layer.attention.output.dense.weight.detach()  # (hidden_dim, all_head_size)
    Wo_heads = [Wo[:, h * head_dim : (h + 1) * head_dim] for h in range(num_heads)]
    return torch.stack(Wo_heads, dim=0)


def get_value_projection_per_head(layer) -> torch.Tensor:
    """
    Extract per-head value projection weights.

    Parameters
    ----------
    layer : BertLayer
        A single BERT encoder layer.

    Returns
    -------
    torch.Tensor
        Wv_heads of shape (num_heads, head_dim, hidden_dim).
    """
    self_attn = layer.attention.self
    num_heads = self_attn.num_attention_heads
    head_dim = self_attn.attention_head_size

    Wv = self_attn.value.weight.detach()  # (all_head_size, hidden_dim)
    return Wv.view(num_heads, head_dim, Wv.shape[1])


def precompute_wov_op_sq_by_layer(model) -> list[np.ndarray]:
    """
    Precompute ||W^o_h W^v_h||_op^2 for every layer and head.

    Parameters
    ----------
    model : BertModel
        BERT-like encoder model.

    Returns
    -------
    list of np.ndarray
        results[layer_idx] is a 1-D array of shape (num_heads,) containing
        ||W^o_h W^v_h||_op^2 for each head.
    """
    results = []

    for layer in model.encoder.layer:
        Wv_heads = get_value_projection_per_head(layer)
        Wo_heads = get_output_projection(layer)

        num_heads = Wv_heads.shape[0]
        op_sq_list = []

        for h in range(num_heads):
            Wv_h = (
                Wv_heads[h].cpu().numpy().astype(np.float64)
            )  # (head_dim, hidden_dim)
            Wo_h = (
                Wo_heads[h].cpu().numpy().astype(np.float64)
            )  # (hidden_dim, head_dim)
            Wov_h = Wo_h @ Wv_h  # (hidden_dim, hidden_dim)
            # ||Wov_h||_op^2 = squared largest singular value
            svals = np.linalg.svd(Wov_h, compute_uv=False)
            op_sq_list.append(float(svals[0] ** 2) if len(svals) > 0 else 0.0)

        results.append(np.array(op_sq_list, dtype=np.float64))

    return results


def manual_attention_block_forward(
    layer,
    hidden_states: torch.Tensor,  # (1, n, hidden_dim)
    extended_mask: torch.Tensor,  # (1, 1, 1, n)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run a manual self-attention block forward pass and verify it matches layer.attention().

    Parameters
    ----------
    layer : BertLayer
        A single BERT encoder layer.
    hidden_states : torch.Tensor
        Shape (1, n, hidden_dim).
    extended_mask : torch.Tensor
        Shape (1, 1, 1, n).

    Returns
    -------
    attn_output_manual : torch.Tensor
        Shape (1, n, hidden_dim).
    attn_probs_manual : torch.Tensor
        Shape (1, num_heads, n, n).
    """
    self_attn = layer.attention.self
    num_heads = self_attn.num_attention_heads
    head_dim = self_attn.attention_head_size

    # ===== actual =====
    attn_outputs = layer.attention(
        hidden_states,
        attention_mask=extended_mask,
        output_attentions=True,
    )
    attn_output_actual = attn_outputs[0]  # (1, n, hidden_dim)

    # ===== manual self-attention =====
    mixed_query_layer = self_attn.query(hidden_states)  # (1, n, all_head_size)
    mixed_key_layer = self_attn.key(hidden_states)  # (1, n, all_head_size)
    mixed_value_layer = self_attn.value(hidden_states)  # (1, n, all_head_size)

    # (batch, seq_len, all_head_size) -> (batch, num_heads, seq_len, head_dim)
    bsz, seq_len, _ = mixed_query_layer.shape
    query_layer = (
        mixed_query_layer.view(bsz, seq_len, num_heads, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    key_layer = (
        mixed_key_layer.view(bsz, seq_len, num_heads, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    value_layer = (
        mixed_value_layer.view(bsz, seq_len, num_heads, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )

    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(head_dim)
    attention_scores = attention_scores + extended_mask

    attn_probs_manual = F.softmax(attention_scores, dim=-1)

    # In eval(), dropout is identity; pass through to match the internal implementation.
    attn_probs_manual = self_attn.dropout(attn_probs_manual)

    context_layer = torch.matmul(attn_probs_manual, value_layer)
    # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, all_head_size)
    context_layer_cat = (
        context_layer.permute(0, 2, 1, 3)
        .contiguous()
        .view(bsz, seq_len, num_heads * head_dim)
    )

    dense_full_manual = layer.attention.output.dense(context_layer_cat)
    attn_output_manual = layer.attention.output.LayerNorm(
        dense_full_manual + hidden_states
    )

    if not torch.allclose(attn_output_manual, attn_output_actual, atol=1e-5, rtol=1e-4):
        diff = (attn_output_manual - attn_output_actual).abs()
        raise AssertionError(
            f"attention_output mismatch: max_abs_diff={diff.max().item():.6e}, "
            f"mean_abs_diff={diff.mean().item():.6e}"
        )

    return attn_output_manual, attn_probs_manual


def manual_layer_forward(
    layer,
    hidden_states: torch.Tensor,  # (1, n, hidden_dim)
    extended_mask: torch.Tensor,  # (1, 1, 1, n)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run a manual BertLayer forward pass and verify it matches layer().

    Parameters
    ----------
    layer : BertLayer
        A single BERT encoder layer.
    hidden_states : torch.Tensor
        Shape (1, n, hidden_dim).
    extended_mask : torch.Tensor
        Shape (1, 1, 1, n).

    Returns
    -------
    layer_output_manual : torch.Tensor
        Shape (1, n, hidden_dim).
    attn_probs_manual : torch.Tensor
        Shape (1, num_heads, n, n).
    """
    # ===== actual =====
    layer_outputs = layer(
        hidden_states,
        attention_mask=extended_mask,
        output_attentions=True,
    )
    layer_output_actual = layer_outputs[0]

    # ===== manual attention block =====
    attn_output_manual, attn_probs_manual = manual_attention_block_forward(
        layer=layer,
        hidden_states=hidden_states,
        extended_mask=extended_mask,
    )

    # ===== manual FFN block =====
    intermediate_linear = layer.intermediate.dense(attn_output_manual)
    intermediate_act = layer.intermediate.intermediate_act_fn(intermediate_linear)
    output_linear = layer.output.dense(intermediate_act)
    layer_output_manual = layer.output.LayerNorm(output_linear + attn_output_manual)

    if not torch.allclose(
        layer_output_manual, layer_output_actual, atol=1e-5, rtol=1e-4
    ):
        diff = (layer_output_manual - layer_output_actual).abs()
        raise AssertionError(
            f"layer_output mismatch: max_abs_diff={diff.max().item():.6e}, "
            f"mean_abs_diff={diff.mean().item():.6e}"
        )

    return layer_output_manual, attn_probs_manual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()

    device = torch.device(
        f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Dataset: {args.input_file}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Device: {device}")

    df = pl.read_csv(args.input_file)
    texts = df["text"].to_list()
    text_ids = df["text_id"].to_list()
    logger.info(f"Number of texts: {len(texts)}")

    logger.info("Building model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_name,
        output_attentions=True,
        output_hidden_states=True,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)

    if not hasattr(model, "encoder") or not hasattr(model.encoder, "layer"):
        raise ValueError(
            "This code assumes a BERT-like encoder model with model.encoder.layer."
        )

    logger.info("Precomputing ||W^o_h W^v_h||_op^2 for all layers and heads...")
    wov_op_sq_by_layer = precompute_wov_op_sq_by_layer(model)
    num_layers = len(model.encoder.layer)

    rows = []
    logger.info("Running manual forward layer-by-layer with asserts...")

    total_batches = (len(texts) + args.batch_size - 1) // args.batch_size

    for start in tqdm(range(0, len(texts), args.batch_size), total=total_batches):
        batch_texts = texts[start : start + args.batch_size]
        batch_text_ids = text_ids[start : start + args.batch_size]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        attention_mask = enc["attention_mask"]

        hidden_states = model.embeddings(
            input_ids=enc.get("input_ids", None),
            token_type_ids=enc.get("token_type_ids", None),
            position_ids=None,
            inputs_embeds=None,
            past_key_values_length=0,
        )

        for b_idx, text_id in enumerate(batch_text_ids):
            seq_len = int(attention_mask[b_idx].sum().item())

            one_mask = attention_mask[b_idx : b_idx + 1, :seq_len]  # (1, n)
            extended_mask = model.get_extended_attention_mask(
                one_mask, one_mask.shape
            ).to(
                device
            )  # (1, 1, 1, n)

            current_hidden = hidden_states[b_idx : b_idx + 1, :seq_len, :]

            for layer_idx in range(num_layers):
                layer = model.encoder.layer[layer_idx]

                next_hidden_manual, attn_probs_manual = manual_layer_forward(
                    layer=layer,
                    hidden_states=current_hidden,
                    extended_mask=extended_mask,
                )

                num_heads = attn_probs_manual.shape[1]
                for head_idx in range(num_heads):
                    A = (
                        attn_probs_manual[0, head_idx]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )  # (n, n)

                    Wov_op_sq = float(wov_op_sq_by_layer[layer_idx][head_idx])

                    if seq_len <= 1:
                        value = 0.0
                    else:
                        PA_fro_sq = compute_pa_fro_sq(A)
                        value = Wov_op_sq * PA_fro_sq / (seq_len - 1)

                    rows.append(
                        {
                            "text_id": text_id,
                            "layer": layer_idx,
                            "head": head_idx,
                            "lambda": value,
                        }
                    )

                current_hidden = next_hidden_manual

    out_df = pl.DataFrame(rows)
    out_df.write_csv(args.output_file)
    logger.info(f"Saved results to {args.output_file}")


if __name__ == "__main__":
    main()
