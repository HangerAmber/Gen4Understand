"""Greedy local VLM inference and parser observations; model downloads are explicit."""

from .io import read_jsonl, write_jsonl


def predict(model_id, manifest, output, adapter=None):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from .training import VideoQACollator

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained(adapter or model_id, max_pixels=256 * 28 * 28)
    processor.tokenizer.padding_side = "right"
    collator = VideoQACollator(processor)
    results = []
    for row in read_jsonl(manifest):
        # Use the same processor/video sampling as adaptation, then remove the dummy answer.
        batch = collator([{**row, "answer": "placeholder", "level": "atom"}])
        labels = batch.pop("labels")
        batch.pop("levels")
        supervised = (labels[0] != -100).nonzero()
        prompt_length = int(supervised[0])
        for key in ("input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids"):
            if key in batch:
                batch[key] = batch[key][:, :prompt_length]
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.inference_mode():
            generated = model.generate(**batch, do_sample=False, max_new_tokens=512)
        answer = processor.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)[0]
        results.append({"id": row["id"], "answer": answer})
    write_jsonl(output, results)
