import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from vits import vit_b_16

class MoCo(nn.Module):
    """
    In global branches, introduce a MoCo_v3 model with a base encoder, a momentum encoder, and two MLPs.(https://arxiv.org/abs/1911.05722)
    In local branches, build a LoDisc model with a shared momentum encoder.(https://arxiv.org/abs/2104.02057)
    """
    def __init__(self, base_encoder, dim=768, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 768)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T

        # build encoders
        self.base_encoder = vit_b_16(pretrained=True)
        self.momentum_encoder = vit_b_16(pretrained=True)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)


        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient
        
        self.patch_size = self.base_encoder.patch_size

        self.relu = nn.ReLU(inplace=True)

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
                mlp.append(nn.Dropout(0.2)) # dropout_rate
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)
  
    def m_extract_attention(self, x, m):
        attentions = []
        self._update_momentum_encoder(m)  # update the momentum encoder
        model = self.momentum_encoder
        with torch.no_grad():  # no gradient
            _ = model(x)
        attentions = [block.self_attention_weights for block in model.encoder.layers]
        return attentions
    
    def fuse_attention(self, attentions_list):
        num = len(attentions_list) 
        last_map = attentions_list[-1]
        for i in range(num-1):
            last_map = torch.mul(attentions_list[i], last_map)
        attmap = last_map[:, 0, 1:]
        return attmap 
    
    def mask_att_idex(self, attmap):
        sort_attmap, idx = torch.sort(attmap, dim=1, descending=True)
        avg_attmap = torch.mean(sort_attmap, dim=1)
        std_attmap = torch.std(sort_attmap, dim=1)

        mask = []
        batch_size, seq_length = attmap.shape[0], attmap.shape[1]
        mask_test = []
        for i in range(batch_size):
            selected_patches = []
            index = []
            threshold = sort_attmap[i, int(seq_length * 0.30)]
            for j in range(seq_length): 
                if attmap[i, j] > threshold:
                    x = 1
                else:
                    x = 0
                index.append(x)
                for k in range(self.patch_size):
                    selected_patches.append(x)
            mask_test.append(index)
            mask.append(selected_patches)
        mask = torch.tensor(mask).to('cuda')
        mask = mask.view(batch_size, int(seq_length ** 0.5), int(seq_length ** 0.5) * self.patch_size) 
        expanded_mask = torch.zeros(batch_size, int(seq_length ** 0.5) * self.patch_size, int(seq_length ** 0.5) * self.patch_size) 
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                for k in range(j * self.patch_size, j * self.patch_size + self.patch_size):
                    expanded_mask[i, k] = mask[i, j]
        return expanded_mask
    
    def mask_input(self, x, mask):
        mask = mask[:, None, :, :]
        mask = mask.repeat(1, 3, 1, 1) 
        mask_x = x.cuda() * mask.cuda()
        return mask_x
    
    def images(self, x1, x2, m):
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))
        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

            q1 = nn.functional.normalize(q1, dim=1)
            q2 = nn.functional.normalize(q2, dim=1)
            k1 = nn.functional.normalize(k1, dim=1)
            k2 = nn.functional.normalize(k2, dim=1)
            # gather all targets
            k1 = concat_all_gather(k1)
            k2 = concat_all_gather(k2)

        # Einstein sum is more intuitive
        logits1 = torch.einsum('nc,mc->nm', [q1, k2]) / self.T
        N1 = logits1.shape[0]  # batch size per GPU
        logits2 = torch.einsum('nc,mc->nm', [q2, k1]) / self.T
        N2 = logits2.shape[0]  # batch size per GPU
        labels1 = (torch.arange(N1, dtype=torch.long) + N1 * torch.distributed.get_rank()).cuda()
        labels2 = (torch.arange(N2, dtype=torch.long) + N2 * torch.distributed.get_rank()).cuda()
        return logits1, labels1, logits2, labels2

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """
        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))
        
        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute global momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)


            # compute momentum attention_mask
            att1 = self.fuse_attention(self.m_extract_attention(x1, m))
            att2 = self.fuse_attention(self.m_extract_attention(x2, m))
            mask_x1 = self.mask_att_idex(att1)
            mask_x2 = self.mask_att_idex(att2)
            input1 = self.mask_input(x1, mask_x1)
            input2 = self.mask_input(x2, mask_x2)
            
            # compute local momentum features as targets
            k_m1 = self.momentum_encoder(input1)
            k_m2 = self.momentum_encoder(input2)

        # compute global loss
        loss1 = self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
        # compute local loss
        loss2 = self.contrastive_loss(q1, k_m2) + self.contrastive_loss(q2, k_m1)
        
        # compute loss
        loss = loss1 + loss2

        return loss
    
    def inference(self, x): 
        out_featmap = self.predictor(self.base_encoder(x)) # [batch_size, hidden_dim]
        out_featmap = nn.functional.normalize(out_featmap, dim=1)
        return out_featmap


class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.hidden_dim
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.hidden_dim
        del self.base_encoder.heads, self.momentum_encoder.heads

        # projectors
        self.base_encoder.heads = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.heads = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
