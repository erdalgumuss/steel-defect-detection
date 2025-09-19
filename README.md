# Steel Defect Detection – Endüstriyel Kalite Kontrol için Yapay Zeka

## 🎯 Proje Hakkında

Bu proje, **Akbank Derin Öğrenme Bootcamp** kapsamında geliştirilmiş; ancak aynı zamanda **gerçek dünya endüstriyel kalite kontrol uygulamalarına referans** olacak şekilde tasarlanmıştır.

Amacımız, çelik üretiminde yüzey kusurlarını otomatik olarak tespit eden bir yapay zekâ pipeline’ı geliştirmek; ve bu süreçte kullanılan **modern mimarileri (CNN, U-Net, Transformer tabanlı segmentasyon, pseudo-labeling)** öğrenip, gelecekte **PET şişirme, taş yünü, cam elyafı** gibi sektörlerde kalite kontrol sistemleri geliştirmeye zemin hazırlamaktır.

---

![Pipeline](image.png)

## 🏗️ Mimari Yaklaşım

Proje, gerçek üretim hattındaki kalite kontrol mantığını taklit eden çok katmanlı bir yapıya sahiptir:

1. **Veri Yönetimi**

   - Kaggle [Severstal: Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection) veri seti kullanıldı.
   - RLE (Run-Length Encoding) maskeleri çözülerek işlenmiş dataset oluşturuldu.
   - Train/Validation splitleri yeniden düzenlendi.

2. **Ön İşleme & EDA**

   - Kusurların boyut, dağılım ve sınıf dengesizlikleri incelendi.
   - Görselleştirmeler ile maskeler görüntüler üzerine bindirildi.

3. **Classifier (Defect var/yok)**

   - EfficientNet tabanlı modeller ile ön tarama yapıldı.
   - Amaç: Kusursuz ürünleri hızlıca eleyerek segmentasyon modelinin yükünü azaltmak.

4. **Segmenter (Defect Localization & Classification)**

   - U-Net ve FPN (EfficientNet/Swin Transformer backbone) mimarileri.
   - Loss: BCE + Dice + Focal kombinasyonları.
   - Augmentations: Flip, Brightness, Cutout/Defect blackout.

5. **Semi-supervised Learning (Pseudo-labeling)**

   - Modelin yüksek güvenli tahminleri eğitim setine dahil edilerek veri zenginleştirildi.

6. **Post-processing**

   - Küçük gürültü maskeleri (<150px) temizlendi.
   - Domain knowledge tabanlı alan eşikleri uygulandı.

7. **Deployment**
   - Gradio tabanlı demo → Görsel yükle, model çıktısını gör.
   - Üretim entegrasyonu için TorchScript / ONNX export hazır.

---

## 📊 Beklenen Kazanımlar

- **Bootcamp**: Güçlü, endüstriyel seviyede bir proje çıktısı.
- **Eğitim**: Modern derin öğrenme mimarilerinde (U-Net, EfficientNet, Transformer, Semi-supervised) deneyim.
- **Gelecek**: Aynı pipeline, başka sektörlere (plastik, cam, taş yünü) kolayca uyarlanabilir.

---

## 🧑‍💻 Teknolojiler

- **PyTorch, segmentation-models-pytorch**
- **Albumentations** (data augmentation)
- **Matplotlib/Seaborn** (EDA & görselleştirme)
- **WandB / TensorBoard** (deney takibi)
- **Gradio / Streamlit** (demo arayüz)

---

## 📂 Repo Yapısı

```
steel-defect-detection/
│── data/ # ham ve işlenmiş veri
│── notebooks/ # EDA ve prototipler
│── src/
│ ├── data/ # dataset, transforms
│ ├── models/ # classifier & segmenter
│ ├── training/ # training loop, metrics, losses
│ ├── inference/ # tahmin, postprocessing
│── scripts/
│ ├── preprocess.py # RLE -> mask, split oluşturma
│ ├── check_dataset.py # görsel kontrol
│ ├── train_baseline.py # baseline eğitim scripti
│── experiments/ # farklı denemeler
│── outputs/ # checkpoint, loss/dice grafikleri, submission
│── requirements.txt # bağımlılıklar
│── Dockerfile # ortam kurulumu
│── README.md # proje açıklaması
```

---

## 📌 Kaynak

- Kaggle Competition: [Severstal: Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection)
- Endüstriyel kalite kontrol literatürü
