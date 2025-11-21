import torch
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from torch import nn
from skimage.metrics import structural_similarity as ssim


def test_model(model, test_dataset, device="cpu", criterion=nn.MSELoss()):
    model.eval()  # Coloca o modelo em modo avaliação

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    test_bar = tqdm(test_loader, desc="🟡 Testando", leave=False)

    test_loss = 0.0
    test_ssim = 0.0
    test_count = 0

    with torch.no_grad():
        for images, _ in test_bar:
            images = images.to(device)
            flat = images.reshape(images.size(0), -1)

            outputs = model(flat)
            outputs = outputs.reshape(-1, 1, 28, 28)

            # MSE
            loss = criterion(outputs.reshape(images.size(0), -1), flat)
            test_loss += loss.item()

            # SSIM
            batch_ssim = 0
            for i in range(images.size(0)):
                img_np = images[i].cpu().numpy().astype("float32")
                out_np = outputs[i].cpu().numpy().astype("float32")
                batch_ssim += ssim(img_np.squeeze(), out_np.squeeze(), data_range=1.0)

            batch_ssim /= images.size(0)
            test_ssim += batch_ssim

            test_count += 1

            test_bar.set_postfix(
                mse=f"{loss.item():.4f}",
                ssim=f"{batch_ssim:.4f}"
            )

    final_test_loss = test_loss / test_count
    final_test_ssim = test_ssim / test_count

    print(f"    📌 Test MSE:  {final_test_loss:.4f}")
    print(f"    📌 Test SSIM: {final_test_ssim:.4f}")

    return final_test_loss, final_test_ssim