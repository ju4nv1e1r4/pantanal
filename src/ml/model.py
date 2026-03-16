import torch
import torch.nn as nn
import timm


class DeepWetlandsModel(nn.Module):
    def __init__(
        self,
        model_name: str = 'efficientnet_b5',
        num_classes: int = 234,
        pretrained: bool = True,
        in_chans: int = 1,
        drop_rate: float = 0.2, # classifier dropout to helps rare classes
    ):
        super().__init__()

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
            drop_rate=drop_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_chans, n_mels, time_frames]
        return self.model(x)

    def param_groups(self):
        head_keywords = ('head', 'classifier', 'fc')
        head_params     = []
        backbone_params = []
        for name, param in self.named_parameters():
            if any(kw in name for kw in head_keywords):
                head_params.append(param)
            else:
                backbone_params.append(param)
        return backbone_params, head_params


if __name__ == "__main__":
    for arch in ('efficientnet_b0','efficientnet_b5', 'convnext_nano'):
        model = DeepWetlandsModel(model_name=arch)
        dummy = torch.randn(8, 1, 128, 313)
        out   = model(dummy)

        backbone_p, head_p = model.param_groups()
        total  = sum(p.numel() for p in model.parameters() if p.requires_grad)
        b_cnt  = sum(p.numel() for p in backbone_p)
        h_cnt  = sum(p.numel() for p in head_p)

        print(f"\n{arch}")
        print(f"  Input : {tuple(dummy.shape)}  →  Output: {tuple(out.shape)}")
        print(f"  Total params : {total:,}")
        print(f"  Backbone     : {b_cnt:,}  |  Head: {h_cnt:,}")
