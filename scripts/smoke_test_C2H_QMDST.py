
import sys
sys.path.insert(0, "..")

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from configs import configs_trainC2H_QMDST as configs

from models.MDSTNet import MDSTNet
from models.Discriminator import (
    MultiDiscriminator,
    MultiDiscriminator_feature
)

from metrics.losses import (
    calc_content_loss,
    calc_style_loss,
    calc_phase_loss
)

from myutils.dataloader import Dataloader


def main():

    device = torch.device(configs.device)

    if device.type != "cuda":
        raise RuntimeError("CUDA GPU is not available!")

    print("=" * 55)
    print("Q-MDST RESEARCH - C2H SMOKE TEST")
    print("=" * 55)

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # --------------------------------------
    # 1. LOAD ONE TRAINING BATCH
    # --------------------------------------

    dataset = Dataloader(
        haze_dir=configs.style_dir,
        clear_dir=configs.content_dir,
        transform=configs.transforms_train,
        unaligned=True
    )

    loader = DataLoader(
        dataset,
        batch_size=configs.batch_size,
        shuffle=True,
        num_workers=0
    )

    batch = next(iter(loader))

    style = batch[0].to(device)
    content = batch[1].to(device)

    print("\n[1] DATA")
    print("Style:", tuple(style.shape))
    print("Content:", tuple(content.shape))

    # --------------------------------------
    # 2. INITIALIZE ORIGINAL C2H MODELS
    # --------------------------------------

    print("\n[2] INITIALIZING MODELS")

    Gen = MDSTNet().to(device)
    D_P = MultiDiscriminator().to(device)
    D_F = MultiDiscriminator_feature().to(device)

    Gen.train()
    D_P.train()
    D_F.train()

    print("Generator: MDSTNet")
    print("Discriminator: MultiDiscriminator")
    print("Feature Discriminator: MultiDiscriminator_feature")

    total_params = sum(
        p.numel() for p in Gen.parameters()
        if p.requires_grad
    )

    print("Generator trainable parameters:", total_params)

    # --------------------------------------
    # 3. OPTIMIZERS
    # --------------------------------------

    opt_Gen = optim.Adam(
        Gen.parameters(),
        lr=configs.lr,
        betas=(0.5, 0.999)
    )

    opt_D_P = optim.Adam(
        D_P.parameters(),
        lr=configs.lr,
        betas=(0.5, 0.999)
    )

    opt_D_F = optim.Adam(
        D_F.parameters(),
        lr=configs.lr,
        betas=(0.5, 0.999)
    )

    mse_loss = nn.MSELoss().to(device)

    label_real = torch.ones(
        [1], dtype=torch.float, device=device
    )

    label_fake = torch.zeros(
        [1], dtype=torch.float, device=device
    )

    # --------------------------------------
    # 4. GENERATOR FORWARD
    # --------------------------------------

    print("\n[3] GENERATOR FORWARD")

    out, out_feats, content_feats, style_feats, \
        c_c, s_s, c_c_feats, s_s_feats = Gen(content, style)

    print("Output:", tuple(out.shape))
    print("Content reconstruction:", tuple(c_c.shape))
    print("Style reconstruction:", tuple(s_s.shape))
    print("Forward: PASSED")

    # --------------------------------------
    # 5. DISCRIMINATOR TRAINING
    # --------------------------------------

    print("\n[4] DISCRIMINATOR BACKWARD")

    loss_d_p = (
        D_P.compute_loss(style, label_real)
        + D_P.compute_loss(out.detach(), label_fake)
    )

    opt_D_P.zero_grad(set_to_none=True)
    loss_d_p.backward()
    opt_D_P.step()

    loss_d_f = (
        D_F.compute_loss(style_feats[2], label_real)
        + D_F.compute_loss(out_feats[2].detach(), label_fake)
    )

    opt_D_F.zero_grad(set_to_none=True)
    loss_d_f.backward()
    opt_D_F.step()

    print("Pixel discriminator loss:", loss_d_p.item())
    print("Feature discriminator loss:", loss_d_f.item())

    # --------------------------------------
    # 6. GENERATOR LOSSES
    # --------------------------------------

    print("\n[5] GENERATOR LOSSES")

    loss_gan_g_p = (
        D_P.compute_loss(out, label_real)
        * configs.gan_p_weight
    )

    loss_gan_g_f = (
        D_F.compute_loss(out_feats[2], label_real)
        * configs.gan_f_weight
    )

    loss_c = (
        calc_content_loss(
            out_feats,
            content_feats,
            norm=True
        ) * configs.content_weight
    )

    loss_s = (
        calc_style_loss(
            out_feats,
            style_feats
        ) * configs.style_weight
    )

    loss_identity_p = (
        mse_loss(c_c, content)
        + mse_loss(s_s, style)
    ) * configs.identity_p_weight

    loss_identity_f = (
        calc_content_loss(c_c_feats, content_feats)
        + calc_content_loss(s_s_feats, style_feats)
    ) * configs.identity_f_weight

    loss_identity_pha = (
        calc_phase_loss(c_c, content, "cos")
        + calc_phase_loss(s_s, style, "cos")
    ) * configs.identity_pha_weight

    total_loss = (
        loss_gan_g_p
        + loss_gan_g_f
        + loss_c
        + loss_s
        + loss_identity_p
        + loss_identity_f
        + loss_identity_pha
    )

    losses = {
        "GAN_pixel": loss_gan_g_p,
        "GAN_feature": loss_gan_g_f,
        "Content": loss_c,
        "Style": loss_s,
        "Identity_pixel": loss_identity_p,
        "Identity_feature": loss_identity_f,
        "Identity_phase": loss_identity_pha,
        "Total": total_loss
    }

    for name, value in losses.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite loss: {name}")

        print(f"{name}: {value.item():.6f}")

    # --------------------------------------
    # 7. GENERATOR BACKWARD
    # --------------------------------------

    print("\n[6] GENERATOR BACKWARD")

    opt_Gen.zero_grad(set_to_none=True)
    total_loss.backward()
    opt_Gen.step()

    print("Generator backward: PASSED")
    print("Optimizer step: PASSED")

    # --------------------------------------
    # 8. GPU MEMORY
    # --------------------------------------

    torch.cuda.synchronize()

    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
    )

    reserved_memory = (
        torch.cuda.max_memory_reserved() / (1024 ** 3)
    )

    print("\n[7] GPU MEMORY")
    print(f"Peak allocated: {peak_memory:.2f} GB")
    print(f"Peak reserved: {reserved_memory:.2f} GB")

    print("\n" + "=" * 55)
    print("C2H SMOKE TEST: PASSED")
    print("No checkpoint was overwritten.")
    print("No training files were modified.")
    print("=" * 55)


if __name__ == "__main__":
    main()
