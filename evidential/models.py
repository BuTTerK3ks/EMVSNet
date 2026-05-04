import torch
import math
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))


class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        # print("Mish activation loaded...")

    def forward(self, x):
        # save 1 second per epoch with no x= x*() and then return x...just inline it.
        return x * (torch.tanh(F.softplus(x)))


def FMish(x):
    '''

    Applies the mish function element-wise:

    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

    See additional documentation for mish class.

    '''

    return x * torch.tanh(F.softplus(x))


def disparity_regression(x, depth_values, max_d=60):
    assert len(x.shape) == 4 or len(x.shape) == 5  # [B, D, H, W] or [B, 1, D, H, W]

    if x.dim() == 5:
        x = x.squeeze(1)  # [B, D, H, W]

    # depth_values: [B, D]
    B, D, H, W = x.shape
    disp_values = depth_values.view(B, D, 1, 1)  # [B, D, 1, 1]
    return torch.sum(x * disp_values, dim=1)  # [B, H, W]


def disparity_classification(x, depth_values):
    assert len(x.shape) == 4
    max_idx = torch.argmax(x, dim=1)
    pred = torch.take(depth_values, max_idx)
    return pred


class HourGlassUp(nn.Module):
    def __init__(self, in_channels):
        super(HourGlassUp, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels * 2, kernel_size=3, stride=2,
                               padding=1, bias=False)

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   Mish())

        # self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
        #                            Mish())
        self.conv3 = nn.Conv3d(in_channels * 2, in_channels * 4, kernel_size=3, stride=2,
                               padding=1, bias=False)

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   Mish())

        # self.conv5 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 2, 1),
        #                            Mish())
        # self.conv5 = nn.Conv3d(in_channels * 4, in_channels * 4, kernel_size=3, stride=2,
        #                        padding=1, bias=False)

        # self.conv6 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
        #                            Mish())
        # self.conv7 = nn.Sequential(
        #     nn.ConvTranspose3d(in_channels * 4, in_channels * 4, 3,
        #                        padding=1, output_padding=1, stride=2, bias=False),
        #     nn.BatchNorm3d(in_channels * 4))

        self.conv8 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3,
                               padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3,
                               padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.combine1 = nn.Sequential(convbn_3d(in_channels * 3, in_channels * 2, 3, 1, 1),
                                      Mish())
        self.combine2 = nn.Sequential(convbn_3d(in_channels * 5, in_channels * 4, 3, 1, 1),
                                      Mish())
        # self.combine3 = nn.Sequential(convbn_3d(in_channels * 6, in_channels * 4, 3, 1, 1),
        #                               Mish())

        self.redir1 = convbn_3d(in_channels, in_channels,
                                kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(
            in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)
        self.redir3 = convbn_3d(
            in_channels * 4, in_channels * 4, kernel_size=1, stride=1, pad=0)

    def forward(self, x, feature4, feature5):
        conv1 = self.conv1(x)  # 1/8
        conv1 = torch.cat((conv1, feature4), dim=1)  # 1/8

        conv1 = self.combine1(conv1)  # 1/8
        conv2 = self.conv2(conv1)  # 1/8

        conv3 = self.conv3(conv2)  # 1/16
        conv3 = torch.cat((conv3, feature5), dim=1)  # 1/16
        conv3 = self.combine2(conv3)  # 1/16
        conv4 = self.conv4(conv3)  # 1/16

        conv7 = FMish(self.redir3(conv4))
        conv8 = FMish(self.conv8(conv7) + self.redir2(conv2))
        conv9 = FMish(self.conv9(conv8) + self.redir1(x))

        return conv9


class HourGlass(nn.Module):
    def __init__(self, in_channels):
        super(HourGlass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   Mish())

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   Mish())

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   Mish())

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   Mish())

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3,
                               padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3,
                               padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels,
                                kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(
            in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = FMish(self.conv5(conv4) + self.redir2(conv2))
        conv6 = FMish(self.conv6(conv5) + self.redir1(x))

        return conv6


class EvidentialWrapper(nn.Module):
    def __init__(self, method='der'):
        super().__init__()
        self.original_model = EvidentialModule(maxdisp=32, method=method).cuda()

    def forward(self, x):
        # Create dummy proj_matrices and depth_values with the expected shape and type
        dummy_depth_values = torch.randn(1, 32).cuda()  # Adjust the shape and values as needed
        return self.original_model(x, dummy_depth_values)


class EvidentialModule(nn.Module):
    def __init__(self, maxdisp, method='der'):
        super(EvidentialModule, self).__init__()

        # ELFNet inspired
        #_______________________________________________________________________________________________
        self.maxdisp = maxdisp
        self.method = method  # 'der' (full NIG) or 'sder' (simplified, alpha = nu + 1)

        self.dres0 = nn.Sequential(convbn_3d(1, 32, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   Mish())

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   Mish())

        self.conv_vol2 = nn.Sequential(convbn_3d(1, 32, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.conv_vol3 = nn.Sequential(convbn_3d(1, 32, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.combine1 = HourGlassUp(32)
        self.dres2 = HourGlass(32)
        self.dres3 = HourGlass(32)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3d(32, 4, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3d(32, 4, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3d(32, 4, kernel_size=3, padding=1, stride=1, bias=False))
        
        # Skip connection projection layers from UNetConvLSTM to evidential processing
        # These project 16-channel 3D skip features to 32-channel to match volume dimensions
        # h0_skip: [B, 16, D, H, W] -> [B, 32, D, H, W] - full resolution, added to volume1/cost0
        # h1_skip: [B, 16, D/2, H/2, W/2] -> [B, 32, D/2, H/2, W/2] - half resolution, added to volume2
        # h2_skip: [B, 16, D/4, H/4, W/4] -> [B, 32, D/4, H/4, W/4] - quarter resolution, added to volume3
        self.skip_proj_h0 = nn.Sequential(
            convbn_3d(16, 32, 3, 1, 1),
            Mish()
        )
        self.skip_proj_h1 = nn.Sequential(
            convbn_3d(16, 32, 3, 1, 1),
            Mish()
        )
        self.skip_proj_h2 = nn.Sequential(
            convbn_3d(16, 32, 3, 1, 1),
            Mish()
        )

    def get_uncertainty(self, logv, logalpha, logbeta):
        v = self.evidence(logv)
        if self.method == 'sder':
            alpha = v + 1.0  # SDER: tie alpha to nu (alpha = nu + 1)
        else:  # 'der'
            alpha = self.evidence(logalpha) + 1  # Independent alpha
        beta = self.evidence(logbeta)
        return v, alpha, beta

    def moe_nig(self, u1, la1, alpha1, beta1, u2, la2, alpha2, beta2):
        # Eq. 9
        la = la1 + la2
        u = (la1 * u1 + u2 * la2) / la
        # u[la == 0] = (u1[la == 0] + u2[la == 0]) * 0.5
        alpha = alpha1 + alpha2 + 0.5
        beta = beta1 + beta2 + 0.5 * \
            (la1 * (u1 - u) ** 2 + la2 * (u2 - u) ** 2)
        return u, la, alpha, beta
 
    def combine_uncertainty(self, ests):
        [u, la, alpha, beta] = ests[0]
        for i in range(1, len(ests)):
            [u1, la1, alpha1, beta1] = ests[i]
            u, la, alpha, beta = self.moe_nig(
                u, la, alpha, beta, u1, la1, alpha1, beta1)
        # SDER (Meinert): each branch has alpha_i = nu_i + 1, and loss_sder / uncertainty_sder
        # assume the predictive simplification alpha = nu + 1 with nu := la (fused precision).
        # Standard MoE NIG fusion uses alpha = alpha1 + alpha2 + 0.5, which breaks that identity
        # (e.g. two experts: la = nu1+nu2 but alpha = nu1+nu2+2.5 instead of nu1+nu2+1).
        if self.method == 'sder':
            alpha = la + 1.0
        return (u, la, alpha, beta)

    def evidence(self, x):
        # return tf.exp(x)
        return F.softplus(x)

    def forward(self, input, depth_value, skip_features=None):
        """
        Forward pass of the evidential module.
        
        Args:
            input: Probability volume [B, 1, D, H, W]
            depth_value: Depth hypotheses [B, D]
            skip_features: Optional dict with skip connections from UNetConvLSTM
                - 'h0': [B, 16, D, H, W] - full resolution features
                - 'h1': [B, 16, D/2, H/2, W/2] - half resolution features
                - 'h2': [B, 16, D/4, H/4, W/4] - quarter resolution features
        
        Returns:
            evidential: Evidential parameters [B, 4, H, W]
            prob_combine: Combined probability volume
        """
        x = input  # already batched: [B, 1, D, H, W]
        
        # Use dynamic depth dimension from depth_value instead of fixed self.maxdisp
        # This allows evaluation with different number of depth hypotheses than training
        assert depth_value.dim() == 2, f"depth_value must have shape [B, D], got {depth_value.shape}"
        num_depth = depth_value.shape[1]

        volume1 = F.interpolate(x, [num_depth, x.size(3), x.size(4)],
                                mode='trilinear', align_corners=True)
        volume1 = F.softmax(volume1, dim=2)

        volume2 = F.interpolate(x, [num_depth // 2, x.size(3) // 2, x.size(4) // 2],
                                mode='trilinear', align_corners=True)
        volume2 = F.softmax(volume2, dim=2)

        volume3 = F.interpolate(x, [num_depth // 4, x.size(3) // 4, x.size(4) // 4],
                                mode='trilinear', align_corners=True)
        volume3 = F.softmax(volume3, dim=2)

        cost0 = self.dres0(volume1)
        cost0 = self.dres1(cost0) + cost0
        
        # Integrate skip features from UNetConvLSTM if provided
        if skip_features is not None:
            # Project h0_skip and add to cost0 (full resolution)
            # h0_skip: [B, 16, D, H, W] -> [B, 32, D, H, W]
            h0_skip = skip_features['h0']
            # Interpolate h0_skip to match cost0 dimensions if needed
            if h0_skip.shape[2:] != cost0.shape[2:]:
                h0_skip = F.interpolate(h0_skip, size=cost0.shape[2:], mode='trilinear', align_corners=True)
            h0_skip_proj = self.skip_proj_h0(h0_skip)
            cost0 = cost0 + h0_skip_proj
            
            # Project h1_skip and add to volume2 before conv_vol2 (half resolution)
            # h1_skip: [B, 16, D/2, H/2, W/2] -> [B, 32, D/2, H/2, W/2]
            h1_skip = skip_features['h1']
            # Interpolate h1_skip to match volume2 dimensions if needed
            if h1_skip.shape[2:] != volume2.shape[2:]:
                h1_skip = F.interpolate(h1_skip, size=volume2.shape[2:], mode='trilinear', align_corners=True)
            h1_skip_proj = self.skip_proj_h1(h1_skip)
            
            # Project h2_skip and add to volume3 before conv_vol3 (quarter resolution)
            # h2_skip: [B, 16, D/4, H/4, W/4] -> [B, 32, D/4, H/4, W/4]
            h2_skip = skip_features['h2']
            # Interpolate h2_skip to match volume3 dimensions if needed
            if h2_skip.shape[2:] != volume3.shape[2:]:
                h2_skip = F.interpolate(h2_skip, size=volume3.shape[2:], mode='trilinear', align_corners=True)
            h2_skip_proj = self.skip_proj_h2(h2_skip)
            
            # Process volume2 and volume3 with skip features added
            volume2 = self.conv_vol2(volume2) + h1_skip_proj
            volume3 = self.conv_vol3(volume3) + h2_skip_proj
        else:
            # Original behavior without skip features
            volume2 = self.conv_vol2(volume2)
            volume3 = self.conv_vol3(volume3)

        combine = self.combine1(cost0, volume2, volume3)
        out1 = self.dres2(combine)
        out2 = self.dres3(out1)

        def get_pred(cost, depth_value):
            cost_upsample = F.interpolate(cost, [num_depth, x.size(3), x.size(4)],
                                          mode='trilinear', align_corners=True)
            cost_upsample = torch.squeeze(cost_upsample, 1)
            prob = F.softmax(cost_upsample, dim=1)
            pred = disparity_regression(prob, depth_value)
            return pred, prob

        def get_logits(cost, prob):
            cost_upsample = F.interpolate(cost, [num_depth, x.size(3), x.size(4)],
                                          mode='trilinear', align_corners=True)
            cost_upsample = torch.squeeze(cost_upsample, 1)
            pred = torch.sum(cost_upsample * prob, 1)
            return pred

        (cost0, logla0, logalpha0, logbeta0) = torch.split(
            self.classif0(cost0), 1, dim=1)
        (cost1, logla1, logalpha1, logbeta1) = torch.split(
            self.classif1(out1), 1, dim=1)
        (cost2, logla2, logalpha2, logbeta2) = torch.split(
            self.classif2(out2), 1, dim=1)

        pred0, prob0 = get_pred(cost0, depth_value=depth_value)
        logla0 = get_logits(logla0, prob0)
        logalpha0 = get_logits(logalpha0, prob0)
        logbeta0 = get_logits(logbeta0, prob0)
        la0, alpha0, beta0 = self.get_uncertainty(
            logla0, logalpha0, logbeta0)

        pred1, prob1 = get_pred(cost1, depth_value=depth_value)
        logla1 = get_logits(logla1, prob1)
        logalpha1 = get_logits(logalpha1, prob1)
        logbeta1 = get_logits(logbeta1, prob1)
        la1, alpha1, beta1 = self.get_uncertainty(
            logla1, logalpha1, logbeta1)

        pred2, prob2 = get_pred(cost2, depth_value=depth_value)
        logla2 = get_logits(logla2, prob2)
        logalpha2 = get_logits(logalpha2, prob2)
        logbeta2 = get_logits(logbeta2, prob2)
        la2, alpha2, beta2 = self.get_uncertainty(
            logla2, logalpha2, logbeta2)

        (u, la, alpha, beta) = self.combine_uncertainty([[pred0, la0, alpha0, beta0], [
            pred1, la1, alpha1, beta1], [pred2, la2, alpha2, beta2]])

        # Stack along dimension 1 (batch dimension) so DataParallel can properly gather along dim=0
        # Shape: [B, 4, H, W] - DataParallel will concatenate batches along dim=0 to get [2*B, 4, H, W]
        evidential = torch.stack((u, la, alpha, beta), dim=1)  # Shape: [B, 4, H, W]
        prob_combine = torch.stack((prob0, prob1, prob2))
        prob_combine = torch.mean(prob_combine, dim=0)

        return evidential, prob_combine

#TODO Make masked
# Original implementation by Amini
def loss_amini_unmasked(gamma, nu, alpha, beta, gt, mask, weight_reg=1.0):

    error = gamma - gt
    omega = 2.0 * beta * (1.0 + nu)

    return torch.mean(
        0.5 * torch.log(math.pi / nu)
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log(error**2 * nu + omega)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
        + weight_reg * torch.abs(error) * (2.0 * nu + alpha)
    )


# Simplified DER (SDER) loss by Meinert - assumes alpha = nu + 1 (alpha unused in formula; must hold for fused la after combine_uncertainty)
def loss_sder(gamma, nu, alpha, beta, gt, mask, weight_reg=1.0):
    eps = 1e-6
    mask = mask.bool()
    error = gamma - gt
    nu_safe = torch.clamp(nu, min=eps)
    var = beta / nu_safe  # SDER variance when alpha = nu + 1
    var = torch.clamp(var, min=eps)

    loss = torch.sum((torch.log(var) + (1. + weight_reg * nu_safe) * error**2 / var)[mask]) / torch.sum(mask)
    return loss


def loss_amini(u, la, alpha, beta, y, mask, weight_reg=1.0):
    eps = 1e-6
    om = 2 * beta * (1 + la)
    mask = mask.bool()
    
    # Clamp values to prevent numerical instability
    la_clamped = torch.clamp(la, min=eps)
    om_clamped = torch.clamp(om, min=eps)
    error_sq = la_clamped * (u - y) ** 2 + om_clamped
    
    loss = torch.sum(
        (0.5 * torch.log(np.pi / la_clamped) - alpha * torch.log(om_clamped) +
         (alpha + 0.5) * torch.log(error_sq) +
         torch.lgamma(alpha) - torch.lgamma(alpha + 0.5))[mask]
    ) / torch.sum(mask == True)

    lossr = weight_reg * (torch.sum((torch.abs(u - y) * (2 * la_clamped + alpha))[mask])) / torch.sum(mask == True)
    loss = loss + lossr

    return loss


# Uncertainty from DER (full NIG formulation)
def uncertainty_der(gamma, nu, alpha, beta):
    eps = 1e-6
    aleatoric = beta / (alpha - 1 + eps)
    epistemic = aleatoric / (nu + eps)
    return aleatoric, epistemic


# Uncertainty from SDER (Meinert) - uses Student-t predictive distribution
def uncertainty_sder(gamma, nu, alpha, beta):
    eps = 1e-6
    aleatoric = torch.sqrt(beta * (1 + nu) / (nu * alpha + eps))
    epistemic = 1. / torch.sqrt(nu + eps)
    return aleatoric, epistemic


def loss_der(outputs, depth_gt, mask, depth_value, method='der', weight_reg=1.0):
    """
    Compute evidential loss for depth estimation.
    
    Args:
        outputs: Dict with 'evidential_prediction' and 'probability_volume'
        depth_gt: Ground truth depth
        mask: Valid pixel mask
        depth_value: Depth hypothesis values
        method: 'der' (full NIG loss) or 'sder' (simplified loss)
        weight_reg: Regularization weight
    
    Returns:
        loss: Computed loss value
        gamma: Predicted depth (mean) - recomputed from probability_volume
        evidential: Dict with all evidential parameters and uncertainties
    """
    evidential_prediction = outputs['evidential_prediction']
    probability_volume = outputs['probability_volume']

    # Recompute gamma from probability_volume using disparity_regression
    # This ensures gamma is consistent with the probability distribution over depth channels
    # probability_volume has shape [B, D, H, W], depth_value has shape [B, D]
    gamma = disparity_regression(probability_volume, depth_value)  # [B, H, W]

    # get EDL parameters (nu, alpha, beta) from evidential_prediction
    # evidential_prediction has shape [B, 4, H, W] from torch.stack (or [2*B, 4, H, W] after DataParallel gathering)
    # Use unbind to properly split along dimension 1 (the 4 parameters) while preserving batch dimension
    _, nu, alpha, beta = torch.unbind(evidential_prediction, dim=1)  # Each: [B, H, W] or [2*B, H, W] after DataParallel
    
    # Select loss based on method
    if method == 'sder':
        loss = loss_sder(gamma, nu, alpha, beta, depth_gt, mask, weight_reg=weight_reg)
        # Compute uncertainties for SDER method
        aleatoric_sder, epistemic_sder = uncertainty_sder(gamma, nu, alpha, beta)
        evidential = {
            'gamma': gamma,
            'nu': nu,
            'alpha': alpha,
            'beta': beta,
            'aleatoric_sder': aleatoric_sder,
            'epistemic_sder': epistemic_sder,
        }
    elif method == 'der':
        loss = loss_amini(gamma, nu, alpha, beta, depth_gt, mask, weight_reg=weight_reg)
        # Compute uncertainties for DER method
        aleatoric_der, epistemic_der = uncertainty_der(gamma, nu, alpha, beta)
        evidential = {
            'gamma': gamma,
            'nu': nu,
            'alpha': alpha,
            'beta': beta,
            'aleatoric_der': aleatoric_der,
            'epistemic_der': epistemic_der,
        }
    else:
        raise ValueError(f"Unknown evidential method: '{method}'. Supported methods are 'der' and 'sder'.")

    return loss, gamma, evidential
