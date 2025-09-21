# scripts/smoke_test.py
import torch
from src.models.unet import UNet
from src.training.losses import BCEDiceLoss
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(in_channels=3, out_channels=4).to(device)
model.train()

x = torch.randn(1,3,256,1600).to(device)   # küçük batch
y_true = torch.randint(0,2,(1,4,256,1600)).float().to(device)  # dummy mask
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = BCEDiceLoss()

optimizer.zero_grad()
logits = model(x)
loss = criterion(logits, y_true)
print("Loss:", loss.item())
loss.backward()
optimizer.step()
print("Backward + optimizer step succeeded")
