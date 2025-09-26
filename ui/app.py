# ui/app.py
import io
import json
import streamlit as st
from PIL import Image

from ui.model_utils import load_model, predict
from ui.viz_utils import overlay_masks
from ui.llm_utils import compute_defect_summary, summarize_with_gpt


def main():
    st.set_page_config(page_title="Steel Defect Segmentation", layout="wide")
    st.title("🔍 Steel Defect Segmentation Demo")
    st.write("Upload an image and the trained model will predict segmentation masks and generate a GPT report.")

    # Modeli yükle
    try:
        model, cfg, device = load_model()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    # Sidebar ayarları
    st.sidebar.header("⚙️ Settings")
    threshold = st.sidebar.slider("Threshold", 0.1, 0.9, 0.5, 0.05)
    alpha = st.sidebar.slider("Mask Transparency", 0.1, 1.0, 0.4, 0.05)
    lang = st.sidebar.radio("Language", ["tr", "en"], index=0)

    # Görsel yükleme
    uploaded_file = st.file_uploader("Upload a steel surface image (.jpg/.png)", type=["jpg", "png"])
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        st.image(pil_img, caption="Uploaded Image", use_column_width=True)

        # Tahmin
        np_img, masks = predict(model, cfg, device, pil_img, threshold=threshold)

        # Overlay
        overlay = overlay_masks(np_img, masks, alpha=alpha)
        st.image(overlay, caption="Prediction Overlay", use_column_width=True)

        # 🔽 İndirme butonu (overlay PNG)
        overlay_pil = Image.fromarray(overlay.astype("uint8"))
        buf = io.BytesIO()
        overlay_pil.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Overlay Image",
            data=buf.getvalue(),
            file_name="overlay.png",
            mime="image/png",
        )

        # Per-class masks
        st.subheader("📌 Per-Class Predictions")
        n_classes = masks.shape[0]
        cols = st.columns(n_classes)
        for i in range(n_classes):
            with cols[i]:
                st.image(masks[i] * 255, caption=f"Class {i+1}", clamp=True)

        # 🔥 GPT Raporu
        st.subheader("🤖 LLM Analysis Report")
        summary = compute_defect_summary(masks)
        summary["language"] = lang  # JSON’a ekledik

        report = summarize_with_gpt(summary, lang=lang)

        # 📊 JSON Summary (daima görünür)
        st.subheader("📊 JSON Summary")
        st.json(summary)

        # 🔽 JSON indirme butonu
        st.download_button(
            label="⬇️ Download JSON Summary",
            data=json.dumps(summary, indent=4, ensure_ascii=False),
            file_name="defect_summary.json",
            mime="application/json",
        )

        # 🤖 GPT Report
        if report and not report.startswith("⚠️"):
            st.markdown(f"### 🤖 GPT Report\n{report}")
        else:
            st.warning(report)


if __name__ == "__main__":
    main()
