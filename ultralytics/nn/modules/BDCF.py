import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv


def AlignResize(x, size, align_corners=False):
    return F.interpolate(
        x, size=size, mode='bilinear', align_corners=align_corners
    )


class GatedContextUnit(nn.Module):
    def __init__(self, align_corners=False, gamma_init=0.1):
        super().__init__()
        self.align_corners = align_corners
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))

    def forward(self, target_feat, self_gate, source_feat, source_gate, comp_gate):
        # Self-enhancement branch
        self_enhanced = target_feat * self_gate

        # Source compensation branch
        source_comp = source_feat * source_gate
        source_comp = AlignResize(
            source_comp,
            size=target_feat.size()[2:],
            align_corners=self.align_corners
        )

        # Align compensation gate to the target resolution
        comp_gate = AlignResize(
            comp_gate,
            size=target_feat.size()[2:],
            align_corners=self.align_corners
        )

        # Residual compensation modulation
        # If comp_gate ≈ 0.5, comp_weight ≈ 1, so the original compensation scale is preserved.
        # If comp_gate > 0.5, compensation is enhanced.
        # If comp_gate < 0.5, compensation is suppressed.
        comp_weight = 1.0 + self.gamma * (2.0 * comp_gate - 1.0)

        # Dynamic compensation fusion
        out = (
            target_feat
            + self_enhanced
            + (1.0 - self_gate) * comp_weight * source_comp
        )

        return out


class BDCF(nn.Module):
    def __init__(self, inc, input_dim=64):
        super().__init__()

        self.input_dim = input_dim
        hidden_dim = input_dim // 2

        # 1. Channel alignment
        # inc[0]: source branch channels
        # inc[1]: target branch channels
        self.source_proj = nn.Conv2d(
            inc[0], hidden_dim, kernel_size=1, bias=False
        )
        self.target_proj = nn.Conv2d(
            inc[1], hidden_dim, kernel_size=1, bias=False
        )

        # 2. Lightweight branch refinement
        self.source_refine = Conv(hidden_dim, hidden_dim, 1)
        self.target_refine = Conv(hidden_dim, hidden_dim, 1)

        # 3. Self-gate generation
        self.gate_act = nn.Sigmoid()

        # 4. Difference-aware compensation gate generation
        # Input: target feature, aligned source feature, and their absolute difference
        self.target_comp_gate = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Sigmoid()
        )

        self.source_comp_gate = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Sigmoid()
        )

        # 5. Gated Context Units
        self.gcu_t = GatedContextUnit(align_corners=False, gamma_init=0.1)
        self.gcu_s = GatedContextUnit(align_corners=False, gamma_init=0.1)

        # 6. Final cross-scale fusion
        self.fuse_conv = Conv(input_dim, input_dim, 3)

    def forward(self, x):
        # The output resolution follows target_feature.
        source_feature, target_feature = x

        # Step 1: channel alignment
        source_aligned = self.source_proj(source_feature)
        target_aligned = self.target_proj(target_feature)

        # Step 2: self-gate generation
        G_s = self.gate_act(source_aligned)
        G_t = self.gate_act(target_aligned)

        # Step 3: lightweight feature refinement
        source_context = self.source_refine(source_aligned)
        target_context = self.target_refine(target_aligned)

        # Step 4: generate compensation gate for target branch
        source_to_target = AlignResize(
            source_context,
            size=target_context.size()[2:],
            align_corners=False
        )
        D_t = torch.abs(target_context - source_to_target)
        C_t = self.target_comp_gate(
            torch.cat([target_context, source_to_target, D_t], dim=1)
        )

        # Step 5: update target branch using source compensation
        target_new = self.gcu_t(
            target_feat=target_context,
            self_gate=G_t,
            source_feat=source_context,
            source_gate=G_s,
            comp_gate=C_t
        )

        # Step 6: generate compensation gate for source branch
        # Here, the updated target branch is used to compensate the source branch,
        # which forms a sequential bidirectional compensation process.
        target_to_source = AlignResize(
            target_new,
            size=source_context.size()[2:],
            align_corners=False
        )
        D_s = torch.abs(source_context - target_to_source)
        C_s = self.source_comp_gate(
            torch.cat([source_context, target_to_source, D_s], dim=1)
        )

        # Step 7: update source branch using the updated target branch
        source_new = self.gcu_s(
            target_feat=source_context,
            self_gate=G_s,
            source_feat=target_new,
            source_gate=G_t,
            comp_gate=C_s
        )

        # Step 8: align source branch to target branch and fuse
        source_new = AlignResize(
            source_new,
            size=target_new.size()[2:],
            align_corners=False
        )

        out = self.fuse_conv(torch.cat([source_new, target_new], dim=1))

        return out