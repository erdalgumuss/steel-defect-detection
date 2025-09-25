# 🏭 Steel Defect Detection – U-Net + ResNet18 Tabanlı Segmentasyon

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Proje Hakkında

Bu proje, **çelik yüzey kusurlarının segmentasyonu** için geliştirilmiş bir **derin öğrenme pipeline** içerir. Amaç, endüstriyel kalite kontrol süreçlerinde kusurların otomatik tespitini sağlamaktır.

Model mimarisi olarak **U-Net** kullanılır ve **ResNet-18 encoder (ImageNet pretrained)** ile desteklenmiştir. Eğitim süreci, modern veri augmentasyonları, Dice + Focal kayıp fonksiyonları ve GPU hızlandırmalı bir PyTorch altyapısı ile gerçekleştirilir.

Bu yapı yalnızca Severstal Steel Defect Detection veriseti için değil, **farklı endüstriyel segmentasyon görevleri** için de kolayca uyarlanabilir.

Dengesiz sınıflar: Focal Loss (küçük sınıflara daha çok önem verir).

Segmentasyon (piksel bazlı maske): Dice Loss (alanların ne kadar çakıştığını ölçer).
Bu yüzden iki adet loss fonksiyonu kullanılmıştır.

---

## ![Pipeline](image.png)

Neden ResNet Encoder?

Feature extraction: ResNet-18, endüstride kanıtlanmış bir mimari. Özellikle skip-connection yapısı sayesinde derin ağlarda vanishing gradient sorununu çözer.

Transfer learning: ImageNet üzerinde eğitilmiş ağırlıkları kullandığımızda, düşük seviye kenar/texture özellikleri daha hızlı öğrenilir. Bu da çelik yüzey kusurlarında avantaj sağlar.

Performans/Verim Deengesi: ResNet-18 hem hafif, hem de güçlüdür → hızlı eğitim, düşük bellek kullanımı, endüstriyel uygulamalara uygunluk.

🔀 Decoder Mode (add vs concat)

add modu: Encoder ve decoder feature map’leri aynnı kanal boyutunda toplanır (up + skip). Daha hafif, daha az parametre → hızlı inference.

concat modu: Encoder ve decoder feature map’leri kanal boyutu boyunca birleştirilir (torch.cat). Daha fazla bilgi taşır ama parametre sayısı ve bellek maliyeti artar.

## Config dosyası üzerinden decoder_mode: "add" | "concat" seçilebilir.

## 📂 Proje Yapısı

```
.
├── Dockerfile
├── README.md
├── config.yaml              # Eğitim ve model ayarları
├── configs/                 # Alternatif config senaryoları
├── docker-compose.yml
├── image.png                # Pipeline görseli
├── notebooks/               # Jupyter notebooklar (keşif, test)
│   └── 01-data-exploration.ipynb
├── requirements.txt         # Gerekli kütüphaneler
└── src/
    ├── config.py            # Config loader (yaml -> dict)
    ├── engines/             # Eğitim ve validasyon döngüleri
    │   ├── training_engine.py
    │   └── evaluation_engine.py
    ├── losses/              # Kayıp fonksiyonları
    │   ├── dice_loss.py
    │   └── focal_loss.py
    ├── main.py              # Ana çalıştırma dosyası (train loop)
    ├── metrics/             # Metrikler
    │   └── dice_coefficient.py
    └── models/              # Model mimarileri
        ├── backbones/       # ResNet, EfficientNet encoderlar
        └── unet.py          # U-Net implementasyonu
```

---

## 🏗️ Pipeline Akışı

### 🔹 1. Veri

- **Dataset**: Kaggle [Severstal: Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection)
- **Format**: RLE maskeler → çok kanallı maskelere dönüştürülür (4 class).
- **Dataset class**: `SteelDefectDataset` (`src/data/dataset.py`)
- **Augmentasyon**: Albumentations kütüphanesi (crop, flip, affine, blur, brightness/contrast, normalize)

### 🔹 2. Model

- **U-Net** + **ResNet-18 encoder** (ImageNet pretrained)
- Encoder-decoder yapısı modüler
- Çok kanallı (4 sınıf) çıkış

### 🔹 3. Loss Fonksiyonları

- **Dice Loss** → Overlap ölçümü
- **Focal Loss** → Class imbalance için odaklanma
- Kombinasyon: Dice + Focal desteklenebilir

### 🔹 4. Metrikler

- **Dice Coefficient** (class-level + mean)
- Notebooklarda görsel test ve loss/metric değerlendirmesi

### 🔹 5. Eğitim Döngüsü

- `train_one_epoch` ve `validate_one_epoch` (src/engines/)
- Adam optimizer, weight decay, learning rate config üzerinden yönetilir
- Checkpoint kaydı ve final model kaydı

---

## ⚙️ Config Yönetimi

Tüm parametreler **config.yaml** üzerinden yönetilir. Örnek:

```yaml
experiment_name: "steel_defect_unet_resnet18"

data:
  train_csv: "data/train.csv"
  train_images_dir: "data/train_images"
  val_split: 0.2
  image_height: 256
  image_width: 1600
  num_classes: 4

training:
  batch_size: 4
  num_workers: 2
  epochs: 20
  learning_rate: 1e-4
  weight_decay: 1e-5
  device: "cuda"

model:
  backbone: "resnet18"
  pretrained: true
  decoder_mode: "add"   # veya "concat"

logging:
  output_dir: "outputs/"
  checkpoint_dir: "checkpoints/"
  save_every: 5
```

---

## 🚀 Kullanım

### 1️⃣ Ortam Kurulumu

```bash
pip install -r requirements.txt
```

veya Docker ile:

```bash
docker build -t steel-defect-detection .
docker run -it steel-defect-detection
```

### 2️⃣ Eğitim

```bash
python src/main.py
```

### 3️⃣ Çıktılar

- `checkpoints/` → her N epoch’ta model checkpointleri
- `outputs/model_final.pth` → final model

### 4️⃣ Notebook Keşfi

```bash
jupyter notebook notebooks/01-data-exploration.ipynb
```

---

## 📊 Özellikler

- ✅ U-Net + ResNet18 encoder
- ✅ Çok kanallı maskeler (4 class)
- ✅ Dice + Focal Loss kombinasyonu
- ✅ Dice metriği (ortalama + class-level)
- ✅ Albumentations ile güçlü augmentasyon
- ✅ Config tabanlı esnek yönetim
- ✅ Docker ile taşınabilirlik

---

## 📌 Kaynaklar

- Kaggle Competition: [Severstal: Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection)
- Endüstriyel yüzey kalite kontrol literatürü
