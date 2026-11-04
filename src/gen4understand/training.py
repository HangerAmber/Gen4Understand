"""Qwen-family SFT: frozen visual backbone, LLM LoRA, trainable merger/projector."""

from .schema import LEVEL_WEIGHTS


def weighted_response_loss(logits, labels, levels):
    """Response-only token NLL with Eq. (5) level weights, applied exactly once."""
    import torch
    from torch.nn import functional as F

    shifted_labels = labels[:, 1:].contiguous()
    losses = F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
        shifted_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )
    valid = shifted_labels != -100
    if not valid.any(1).all():
        raise ValueError("Every example must contain supervised response tokens")
    per_example = (losses.view_as(shifted_labels) * valid).sum(1) / valid.sum(1)
    weights = torch.tensor([LEVEL_WEIGHTS[x] for x in levels], device=logits.device)
    return (per_example * weights).mean()


def attach_lora(model, rank=64, alpha=128):
    from peft import LoraConfig, get_peft_model
    from torch import nn

    # Match complete module names to prevent accidental LoRA injection into ViT.
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and "language_model" in name
        and not name.endswith("lm_head")
    ]
    projectors = [
        name
        for name, _ in model.named_modules()
        if name.endswith("visual.merger")
        or (
            "visual" in name
            and ".deepstack_merger_list." in name
            and name.rsplit(".", 1)[-1].isdigit()
        )
    ]
    if not targets or not projectors:
        raise ValueError(
            "Unsupported module layout: inspect LLM/projector names before adapting this backbone"
        )
    model.requires_grad_(False)
    model = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=targets,
            modules_to_save=projectors,
            lora_dropout=0.0,
            task_type="CAUSAL_LM",
        ),
    )
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and "visual" in name and "merger" not in name:
            raise RuntimeError(f"Visual encoder unexpectedly trainable: {name}")
    return model


class VideoQACollator:
    """Mask prompt/padding and reject overflow instead of truncating visual tokens."""

    def __init__(self, processor, max_length=4096, frames=16, max_side=1280):
        self.processor = processor
        self.max_length = max_length
        self.frames = frames
        self.max_side = max_side

    def __call__(self, records):
        import numpy as np
        import torch
        from PIL import Image

        from .video import read_video

        videos, full_texts, prompt_texts, sampled_fps = [], [], [], []
        for row in records:
            frames, source_fps = read_video(row["video"])
            sampled_fps.append(self.frames * source_fps / len(frames))
            frames = frames[np.linspace(0, len(frames) - 1, self.frames).round().astype(int)]
            height, width = frames.shape[1:3]
            factor = min(1.0, self.max_side / max(height, width))
            if factor < 1:
                size = (max(1, round(width * factor)), max(1, round(height * factor)))
                frames = np.stack(
                    [np.asarray(Image.fromarray(frame).resize(size)) for frame in frames]
                )
            videos.append(frames)
            user = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": row["video"]},
                        {"type": "text", "text": row["question"]},
                    ],
                }
            ]
            prompt_texts.append(
                self.processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
            )
            full_texts.append(
                self.processor.apply_chat_template(
                    user + [{"role": "assistant", "content": row["answer"]}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        # Qwen2.5 processor expands video tokens from the supplied frame arrays.
        batch = self.processor(
            text=full_texts, videos=videos, fps=sampled_fps, padding=True, return_tensors="pt"
        )
        prompts = self.processor(
            text=prompt_texts, videos=videos, fps=sampled_fps, padding=True, return_tensors="pt"
        )
        if batch["input_ids"].shape[1] > self.max_length:
            raise ValueError(
                "Visual/text tokens exceed max_length; lower processor pixel budget or split the clip"
            )
        labels = batch["input_ids"].clone()
        for i in range(len(records)):
            n = int(prompts["attention_mask"][i].sum())
            if not torch.equal(batch["input_ids"][i, :n], prompts["input_ids"][i, :n]):
                raise ValueError("Chat template does not preserve the prompt prefix")
            labels[i, :n] = -100
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        batch["levels"] = [row["level"] for row in records]
        return batch


def train(config_path, manifest, output, resume=None):
    import torch
    import yaml
    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

    from .io import read_jsonl

    config = yaml.safe_load(open(config_path, encoding="utf-8"))
    rows = list(read_jsonl(manifest))
    if not rows or any(r["split"] != "train" for r in rows):
        raise ValueError("SFT requires nonempty training-only examples")
    if not torch.cuda.is_available():
        raise RuntimeError("Full VLM SFT requires a CUDA GPU; run g4u demo for the CPU pipeline")
    if config["bf16"] and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU does not support the requested BF16 precision")
    torch.manual_seed(config["seed"])
    model = AutoModelForImageTextToText.from_pretrained(
        config["model"], torch_dtype=torch.bfloat16 if config["bf16"] else torch.float32
    )
    model = attach_lora(model, config["lora_rank"], config["lora_alpha"])
    model.config.use_cache = False
    model.enable_input_require_grads()
    processor = AutoProcessor.from_pretrained(config["model"], max_pixels=config["max_pixels"])
    processor.tokenizer.padding_side = "right"

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            levels = inputs.pop("levels")
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = weighted_response_loss(outputs.logits, labels, levels)
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=output,
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation"],
        learning_rate=config["learning_rate"],
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=config["bf16"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        seed=config["seed"],
        dataloader_num_workers=0,
    )
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=rows,
        data_collator=VideoQACollator(
            processor, config["max_length"], config["frames"], config["max_side"]
        ),
    )
    # Custom loss is already averaged per microbatch, so Trainer must scale it
    # by gradient accumulation rather than assuming we used num_items_in_batch.
    trainer.model_accepts_loss_kwargs = False
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(output)
    processor.save_pretrained(output)
