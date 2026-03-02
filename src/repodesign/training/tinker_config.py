"""Training configuration for Tinker (full LoRA) with Qwen3-VL-235B-A22B-Instruct.

This module defines the training configurations for both SFT and GRPO stages.
Actual training runs happen on rented GPU infrastructure (A100/H100).
This config is consumed by the training scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    """LoRA adapter configuration for Tinker."""

    rank: int = 64
    alpha: int = 128  # 2x rank is standard
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class SFTConfig:
    """Stage 1: Supervised Fine-Tuning configuration.

    Two SFT tasks:
    1. Repo → IR extraction: Given a repo's directory tree + code snippets,
       generate the structured Repo IR JSON.
    2. Spec + IR → Plan: Given a spec and Repo IR, generate an
       implementation plan with real file paths.
    """

    model_name: str = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # Training hyperparameters
    num_epochs: int = 3
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 32  # Effective batch = 32
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_seq_length: int = 8192  # Qwen3-VL supports up to 32k
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Data
    dataset_path: str = "data/train/sft_data.jsonl"
    eval_dataset_path: str = "data/train/sft_eval.jsonl"

    # Output
    output_dir: str = "output/sft_checkpoints"
    save_steps: int = 200
    save_total_limit: int = 3
    eval_steps: int = 100
    logging_steps: int = 10

    # Packing (combine short examples for efficiency)
    packing: bool = True


@dataclass
class GRPOConfig:
    """Stage 2: Group Relative Policy Optimization configuration.

    GRPO over contrastive scale-tier pairs:
    - Preferred: Plans that match the stated scale tier
    - Rejected: Plans that over-engineer or under-engineer

    GRPO generates multiple outputs per prompt, scores them with a reward
    function, and optimizes using group-based advantages (no critic model needed).
    """

    model_name: str = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    sft_checkpoint: str = "output/sft_checkpoints/final"
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # GRPO-specific
    num_generations: int = 4  # Generate N responses per prompt
    beta: float = 0.1  # KL penalty coefficient
    reward_weights: dict = field(default_factory=lambda: {
        "repo_grounding_score": 0.4,
        "scale_appropriateness": 0.3,
        "plan_completeness": 0.2,
        "format_compliance": 0.1,
    })

    # Training hyperparameters
    num_epochs: int = 1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    max_seq_length: int = 8192
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Data
    dataset_path: str = "data/train/grpo_data.jsonl"

    # Output
    output_dir: str = "output/grpo_checkpoints"
    save_steps: int = 100
    save_total_limit: int = 3
    logging_steps: int = 5


@dataclass
class TrainingPipeline:
    """Full training pipeline configuration."""

    sft: SFTConfig = field(default_factory=SFTConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

    # Hardware requirements
    gpu_type: str = "A100-80GB"  # or H100
    num_gpus: int = 4  # For 235B model with LoRA
    estimated_sft_hours: float = 12.0
    estimated_grpo_hours: float = 8.0

    def get_estimated_cost(self, price_per_gpu_hour: float = 2.0) -> dict:
        """Estimate training cost."""
        sft_cost = self.num_gpus * self.estimated_sft_hours * price_per_gpu_hour
        grpo_cost = self.num_gpus * self.estimated_grpo_hours * price_per_gpu_hour
        return {
            "sft_cost_usd": sft_cost,
            "grpo_cost_usd": grpo_cost,
            "total_cost_usd": sft_cost + grpo_cost,
        }
