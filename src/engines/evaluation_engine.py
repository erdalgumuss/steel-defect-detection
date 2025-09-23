import torch
from tqdm import tqdm

def evaluate_one_epoch(model, dataloader, metric, device):
    """
    Modeli bir epoch boyunca değerlendiren fonksiyon.

    Args:
        model (torch.nn.Module): Değerlendirilecek model.
        dataloader (torch.utils.data.DataLoader): Doğrulama veri yükleyicisi.
        metric (callable): Kullanılacak değerlendirme metriği (örn. Dice Katsayısı).
        device (torch.device): Kullanılacak cihaz (CPU veya GPU).

    Returns:
        float: Epoch için ortalama metrik skoru.
    """
    model.eval()  # Modeli değerlendirme moduna al
    total_score = 0.0
    
    with torch.no_grad():  # Gradyan hesaplamasını devre dışı bırak
        for images, masks in tqdm(dataloader, desc="Doğrulama"):
            images = images.to(device)
            masks = masks.to(device)
            
            # İleri besleme (forward pass)
            outputs = model(images)
            
            # Tahminleri olasılık değerlerinden maskelere dönüştür
            predicted_masks = (outputs > 0.5).float() # Eşikleme ile ikili tahmin maskesi oluştur
            
            # Metrik skorunu hesapla ve topla
            total_score += metric(predicted_masks, masks).item()
            
    avg_score = total_score / len(dataloader)
    return avg_score