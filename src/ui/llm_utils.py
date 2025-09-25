# src/ui/llm_utils.py
import os
import json
import numpy as np
import cv2
from dotenv import load_dotenv

# --- .env yükleme ---
load_dotenv()

# --- OpenAI Client ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
CLIENT = None
if OPENAI_API_KEY:
    from openai import OpenAI
    CLIENT = OpenAI(api_key=OPENAI_API_KEY)


# ---------------------------
# Defect Summary (numerical)
# ---------------------------
def compute_defect_summary(masks: np.ndarray) -> dict:
    """
    Compute pixel-level summary statistics for each defect class.

    Args:
        masks (np.ndarray): Binary masks of shape (C, H, W)

    Returns:
        dict: JSON-like summary with class-level info
    """
    C, H, W = masks.shape
    total = H * W
    summary = {
        "image_height": H,
        "image_width": W,
        "total_pixels": int(total),
        "defects": []
    }

    for c in range(C):
        m = (masks[c] > 0).astype(np.uint8)
        area = int(m.sum())
        pct = float(area / total) if total > 0 else 0.0
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        components = max(0, num_labels - 1)

        largest = None
        if components > 0:
            comp_stats = stats[1:]  # skip background
            idx = np.argmax(comp_stats[:, cv2.CC_STAT_AREA])
            x, y, w, h, a = comp_stats[idx, :5]
            largest = {
                "bbox_xywh": [int(x), int(y), int(w), int(h)],
                "area_pixels": int(a),
                "area_percentage": float(a / total) if total > 0 else 0.0
            }

        summary["defects"].append({
            "class_id": c + 1,
            "pixel_area": area,
            "area_percentage": pct,
            "components": components,
            "largest_component": largest
        })

    return summary


# ---------------------------
# GPT Summarization
# ---------------------------
def summarize_with_gpt(defect_json: dict, model_name: str = "gpt-4o-mini", lang: str = "tr") -> str:
    if CLIENT is None:
        return "⚠️ LLM devre dışı: OPENAI_API_KEY bulunamadı."

    # Prompts
    if lang == "tr":
        system_prompt = (
            "Sen bir kalite kontrol asistanısın. "
            "Verilen JSON formatındaki hata segmentasyon sonuçlarını "
            "incele ve kullanıcıya anlaşılır bir özet rapor hazırla. "
            "Her sınıf için hata oranlarını, bileşen sayılarını ve büyük kusurları "
            "net ve kısa cümlelerle Türkçe olarak anlat."
        )
    else:
        system_prompt = (
            "You are a quality control assistant. "
            "You will read the defect segmentation results in JSON format "
            "and provide a clear natural language summary. "
            "Explain per-class defect ratios, component counts, and largest defects "
            "in short, clear sentences in English."
        )

    user_prompt = f"{json.dumps(defect_json, ensure_ascii=False, indent=2)}"
     # 🔍 Debug: log to console
    print("=== GPT API PAYLOAD ===")
    print("Model:", model_name)
    print("System prompt:", system_prompt)
    print("User prompt:", user_prompt)
    print("=======================")

    try:
        resp = CLIENT.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ GPT isteğinde hata oluştu: {str(e)}"
