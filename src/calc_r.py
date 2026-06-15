import argparse
import math
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from utils.mylogging import init_logger
from utils.seed import fix_seeds

logger = init_logger(__name__, log_file_path=Path("logs/calc_r.log"))
fix_seeds(0)


def compute_r_from_H_and_Z(
    hidden_states: torch.Tensor,
    z_states: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """
    Compute r = numerator / denominator where
        numerator   = (1/n) sum_j ||h_j - mu(H)||^2
        denominator = ||mu(H + Z)||^2

    Parameters
    ----------
    hidden_states : torch.Tensor
        Shape (1, n, d) = H.
    z_states : torch.Tensor
        Shape (1, n, d) = Z.
    eps : float
        Clamp for denominator.

    Returns
    -------
    float
        r value.
    """
    H = hidden_states[0]  # (n, d)
    Z = z_states[0]

    mu_H = H.mean(dim=0, keepdim=True)  # (1, d)
    numerator = (H - mu_H).pow(2).sum(dim=1).mean()  # scalar

    mu_residual = (H + Z).mean(dim=0)  # (d,)
    denominator = mu_residual.pow(2).sum()  # scalar

    return (numerator / denominator.clamp_min(eps)).item()


def manual_attention_block_forward(
    layer,
    hidden_states: torch.Tensor,  # (1, n, hidden_dim)
    extended_mask: torch.Tensor,  # (1, 1, 1, n)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        Shape (1, n, hidden_dim) = LayerNorm(H + Z).
    attn_probs_manual : torch.Tensor
        Shape (1, num_heads, n, n).
    z_states_manual : torch.Tensor
        Shape (1, n, hidden_dim) = Z.
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

    z_states_manual = layer.attention.output.dense(context_layer_cat)  # (1, n, d) = Z
    attn_output_manual = layer.attention.output.LayerNorm(
        z_states_manual + hidden_states
    )

    if not torch.allclose(attn_output_manual, attn_output_actual, atol=1e-5, rtol=1e-4):
        diff = (attn_output_manual - attn_output_actual).abs()
        raise AssertionError(
            f"attention_output mismatch: max_abs_diff={diff.max().item():.6e}, "
            f"mean_abs_diff={diff.mean().item():.6e}"
        )

    return attn_output_manual, attn_probs_manual, z_states_manual


def manual_layer_forward(
    layer,
    hidden_states: torch.Tensor,  # (1, n, hidden_dim)
    extended_mask: torch.Tensor,  # (1, 1, 1, n)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    z_states_manual : torch.Tensor
        Shape (1, n, hidden_dim) = attention projection output Z.
    """
    # ===== actual =====
    layer_outputs = layer(
        hidden_states,
        attention_mask=extended_mask,
        output_attentions=True,
    )
    layer_output_actual = layer_outputs[0]

    # ===== manual attention block =====
    attn_output_manual, attn_probs_manual, z_states_manual = (
        manual_attention_block_forward(
            layer=layer,
            hidden_states=hidden_states,
            extended_mask=extended_mask,
        )
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

    return layer_output_manual, attn_probs_manual, z_states_manual


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

                H = current_hidden

                next_hidden_manual, _, Z = manual_layer_forward(
                    layer=layer,
                    hidden_states=current_hidden,
                    extended_mask=extended_mask,
                )

                rows.append(
                    {
                        "text_id": text_id,
                        "layer": layer_idx,
                        "r": compute_r_from_H_and_Z(H, Z),
                    }
                )

                current_hidden = next_hidden_manual

    out_df = pl.DataFrame(rows)
    out_df.write_csv(args.output_file)
    logger.info(f"Saved results to {args.output_file}")


if __name__ == "__main__":
    main()
