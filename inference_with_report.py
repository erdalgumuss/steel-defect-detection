import os
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from config import Config
from data.transforms import get_infer_transforms
from models.unet import UNetResNet18

# (Opsiyonel) OpenAI entegrasyonu
from openai import OpenAI
client = OpenAI()

@torch.no_grad()
def infer_single_image(model, image_path, device, cfg):
    """Tek görsel için maske çıkarır."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = np.repeat(img[..., None], 3, axis=2)
    tfms = get_infer_transforms(cfg["data"]["image_height"], cfg["data"]["image_width"])
    sample = tfms(image=img)
    tensor = sample["image"].unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.sigmoid(logits)
    mask = (probs > 0.5).float().cpu().numpy()[0]  # (C,H,W)
    return mask

def summarize_mask(mask: np.ndarray) -> dict:
    """Maskeden özet çıkarır (alan yüzdesi)."""
    summaries = {}
    H, W = mask.shape[1], mask.shape[2]
    total_pixels = H * W
    for c in range(mask.shape[0]):
        area = mask[c].sum()
        summaries[f"class_{c+1}"] = {
            "area_pixels": int(area),
            "area_percent": float(area / total_pixels * 100.0)
        }
    return summaries

def generate_report(summary: dict, image_id: str) -> str:
    """LLM API çağrısı ile rapor üretir."""
    prompt = f"""
    Aşağıdaki çelik yüzey segmentasyon sonuçlarını analiz et ve kalite kontrol raporu hazırla.
    
    Görsel: {image_id}
    Sonuçlar: {json.dumps(summary, indent=2)}
    
    Açıklayıcı bir rapor ver, hataları açıkla ve kalite derecesi öner.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # ihtiyaca göre değiştirilebilir
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content

def main(config_path="config.yaml", checkpoint="outputs/model_final.pth", infer_dir="data/test_images"):
    cfg = Config(config_path).dict
    device = cfg["training"]["device"] if torch.cuda.is_available() else "cpu"

    # Model yükle
    model = UNetResNet18(
        num_classes=cfg["data"]["num_classes"],
        pretrained=False,
        decoder_mode=cfg["model"].get("decoder_mode", "add")
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    os.makedirs("inference_outputs", exist_ok=True)

    results = []
    for img_name in os.listdir(infer_dir):
        img_path = os.path.join(infer_dir, img_name)
        mask = infer_single_image(model, img_path, device, cfg)

        # Maskeden özet çıkar
        summary = summarize_mask(mask)

        # Rapor üret
        report = generate_report(summary, img_name)

        # JSON kayıt
        results.append({
            "image_id": img_name,
            "summary": summary,
            "report": report
        })

        # PNG olarak kaydet
        for c in range(cfg["data"]["num_classes"]):
            out_path = f"inference_outputs/{img_name}_class{c+1}.png"
            cv2.imwrite(out_path, (mask[c] * 255).astype(np.uint8))

    # Toplu sonuç kaydet
    with open("inference_outputs/results.json", "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("Inference + raporlar kaydedildi → inference_outputs/results.json")

if __name__ == "__main__":
    main()
