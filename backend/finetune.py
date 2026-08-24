"""Fine-tune TrOCR on handwritten prescription words.

Stock TrOCR reads doctors' handwriting at ~63% character error - unusable. This
trains the recognizer on real prescription words and writes a checkpoint that
ocr.py picks up automatically.

    python backend/finetune.py /path/to/rxhandbd --epochs 3

Data: RxHandBD (https://zenodo.org/records/18478741), MIT-licensed. Train split
only; the test split is never touched here so the score means something.
"""
import argparse
import csv
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


BASE = Path(__file__).resolve().parent
DEFAULT_OUT = BASE / "models" / "trocr-rx"


class WordImages(Dataset):
    def __init__(self, rows, folder, processor, max_len=24):
        self.rows, self.folder, self.proc, self.max_len = rows, folder, processor, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        image = Image.open(self.folder / row["Images"]).convert("RGB")
        pixels = self.proc(images=image, return_tensors="pt").pixel_values[0]
        labels = self.proc.tokenizer(row["Text"].strip(), padding="max_length",
                                     max_length=self.max_len, truncation=True).input_ids
        # -100 tells the loss to ignore padding.
        labels = [t if t != self.proc.tokenizer.pad_token_id else -100 for t in labels]
        return {"pixel_values": pixels, "labels": torch.tensor(labels)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path, help="unzipped RxHandBD folder")
    ap.add_argument("--model", default="microsoft/trocr-base-handwritten")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--val", type=int, default=200, help="images held out of training for CER")
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (for timing runs)")
    args = ap.parse_args()

    rows = list(csv.DictReader((args.dataset / "Train_Label.csv").open()))
    random.seed(0)
    random.shuffle(rows)
    folder = args.dataset / "Train_Set"
    val_rows, rows = rows[:args.val], rows[args.val:]

    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()

    loader = DataLoader(WordImages(rows, folder, processor), batch_size=args.batch,
                        shuffle=True, num_workers=2)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def validation_cer():
        """Character error rate on held-out training images - the honest progress signal."""
        model.eval()
        errors = characters = 0
        with torch.no_grad():
            for i in range(0, len(val_rows), args.batch):
                chunk = val_rows[i:i + args.batch]
                images = [Image.open(folder / r["Images"]).convert("RGB") for r in chunk]
                pixels = processor(images=images, return_tensors="pt").pixel_values.to(device)
                decoded = processor.batch_decode(model.generate(pixels, max_new_tokens=24),
                                                 skip_special_tokens=True)
                for row, prediction in zip(chunk, decoded):
                    truth = row["Text"].strip().lower()
                    errors += edit_distance(truth, prediction.strip().lower())
                    characters += len(truth)
        model.train()
        return errors / max(characters, 1)

    step = 0
    best = 1e9
    started = time.time()
    for epoch in range(args.epochs):
        running = 0.0
        for batch in loader:
            loss = model(pixel_values=batch["pixel_values"].to(device),
                         labels=batch["labels"].to(device)).loss
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            running += loss.item()
            step += 1
            if step % 20 == 0:
                rate = step * args.batch / (time.time() - started)
                print(f"epoch {epoch + 1} step {step} loss {running / 20:.4f} "
                      f"({rate:.1f} img/s)", flush=True)
                running = 0.0
            if args.max_steps and step >= args.max_steps:
                print(f"stopping at {step} steps (timing run)")
                return

        cer = validation_cer()
        print(f"== epoch {epoch + 1}: val CER {cer:.1%} "
              f"({(time.time() - started) / 60:.1f} min elapsed)", flush=True)
        if cer < best:            # keep the best epoch, not the last one
            best = cer
            args.out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.out)
            processor.save_pretrained(args.out)
            print(f"   saved checkpoint to {args.out}", flush=True)

    print(f"done in {(time.time() - started) / 60:.1f} min, best val CER {best:.1%}")


if __name__ == "__main__":
    main()
