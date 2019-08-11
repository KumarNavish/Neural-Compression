import math
from collections import OrderedDict

import torch.nn.functional as F

import torch
from torch import nn as nn
from torch.nn import Parameter, Sequential
from utee import misc
from torch.nn.modules.utils import _pair

import scipy.linalg as sl


print = misc.logger.info

# using hardware parameters from Eyeriss

default_s1 = int(100 * 1024 / 2)  # input cache, 100K (16-bit Fixed-Point)
default_s2 = 1 * int(8 * 1024 / 2)  # kernel cache, 8K (16-bit Fixed-Point)
default_m = 12
default_n = 14

# unit energy constants
default_e_mac = 1.0 + 1.0 + 1.0  # including both read and write RF
default_e_mem = 200.0
default_e_cache = 6.0
default_e_rf = 1.0


class Layer_energy(object):
    def __init__(self, **kwargs):
        super(Layer_energy, self).__init__()
        self.h = kwargs['h'] if 'h' in kwargs else None
        self.w = kwargs['w'] if 'w' in kwargs else None
        self.c = kwargs['c'] if 'c' in kwargs else None
        self.d = kwargs['d'] if 'd' in kwargs else None
        self.xi = kwargs['xi'] if 'xi' in kwargs else None
        self.g = kwargs['g'] if 'g' in kwargs else None
        self.p = kwargs['p'] if 'p' in kwargs else None
        self.m = kwargs['m'] if 'm' in kwargs else None
        self.n = kwargs['n'] if 'n' in kwargs else None
        self.s1 = kwargs['s1'] if 's1' in kwargs else None
        self.s2 = kwargs['s2'] if 's2' in kwargs else None
        self.r = kwargs['r'] if 'r' in kwargs else None
        self.is_conv = True if self.r is not None else False

        if self.h is not None:
            self.h_ = max(0.0, math.floor((self.h + 2.0 * self.p - self.r) / float(self.xi)) + 1)
        if self.w is not None:
            self.w_ = max(0.0, math.floor((self.w + 2.0 * self.p - self.r) / float(self.xi)) + 1)

        self.cached_Xenergy = None

    def get_alpha(self, e_mem, e_cache, e_rf):
        if self.is_conv:
            return e_mem + \
                   (math.ceil((float(self.d) / self.g) / self.n) * (self.r ** 2) / float(self.xi ** 2)) * e_cache + \
                   ((float(self.d) / self.g) * (self.r ** 2) / (self.xi ** 2)) * e_rf
        else:
            if self.c <= default_s1:
                return e_mem + math.ceil(float(self.d) / self.n) * e_cache + float(self.d) * e_rf
            else:
                return math.ceil(float(self.d) / self.n) * e_mem + math.ceil(float(self.d) / self.n) * e_cache + float(
                    self.d) * e_rf

    def get_beta(self, e_mem, e_cache, e_rf, in_cache=None):
        if self.is_conv:
            n = 1 if in_cache else math.ceil(self.h_ * self.w_ / self.m)
            return n * e_mem + math.ceil(self.h_ * self.w_ / self.m) * e_cache + \
                   (self.h_ * self.w_) * e_rf
        else:
            return e_mem + e_cache + e_rf

    def get_gamma(self, e_mem, k=None):
        if self.is_conv:
            rows_per_batch = math.floor(self.s1 / float(k))
            assert rows_per_batch >= self.r
            # print(self.__dict__)
            # print('###########', rows_per_batch, self.s1, k)
            # print('conv input data energy (2):{:.2e}'.format(float(k) * (self.r - 1) * (math.ceil(float(self.h) / (rows_per_batch - self.r + 1)) - 1)))

            return (float(self.d) * self.h_ * self.w_) * e_mem + \
                   float(k) * (self.r - self.xi) * \
                   max(0.0, (math.ceil(float(self.h) / (rows_per_batch - self.r + self.xi)) - 1)) * e_mem
        else:
            return float(self.d) * e_mem

    def get_knapsack_weight_W(self, e_mac, e_mem, e_cache, e_rf, in_cache=None, crelax=False):
        if self.is_conv:
            if crelax:
                # use relaxed computation energy estimation (larger than the real computation energy)
                return self.get_beta(e_mem, e_cache, e_rf, in_cache) + e_mac * self.h_ * self.w_
            else:
                # computation energy will be included in other place
                return self.get_beta(e_mem, e_cache, e_rf, in_cache) + e_mac * 0.0
        else:
            return self.get_beta(e_mem, e_cache, e_rf, in_cache) + e_mac

    def get_knapsack_bound_W(self, e_mem, e_cache, e_rf, X_nnz, k):
        if self.is_conv:
            return self.get_gamma(e_mem, k) + self.get_alpha(e_mem, e_cache, e_rf) * X_nnz
        else:
            return self.get_gamma(e_mem) + self.get_alpha(e_mem, e_cache, e_rf) * X_nnz


def build_energy_info(model, m=default_m, n=default_n, s1=default_s1, s2=default_s2):
    res = {}
    for name, p in model.named_parameters():
        if name.endswith('input_mask'):
            layer_name = name[:-len('input_mask') - 1]
            if layer_name in res:
                res[layer_name]['h'] = p.size()[1]
                res[layer_name]['w'] = p.size()[2]
            else:
                res[layer_name] = {'h': p.size()[1], 'w': p.size()[2]}
        elif name.endswith('.hw'):
            layer_name = name[:-len('hw') - 1]
            if layer_name in res:
                res[layer_name]['h'] = float(p.data[0])
                res[layer_name]['w'] = float(p.data[1])
            else:
                res[layer_name] = {'h': float(p.data[0]), 'w': float(p.data[1])}
        elif name.endswith('.xi'):
            layer_name = name[:-len('xi') - 1]
            if layer_name in res:
                res[layer_name]['xi'] = float(p.data[0])
            else:
                res[layer_name] = {'xi': float(p.data[0])}
        elif name.endswith('.g'):
            layer_name = name[:-len('g') - 1]
            if layer_name in res:
                res[layer_name]['g'] = float(p.data[0])
            else:
                res[layer_name] = {'g': float(p.data[0])}
        elif name.endswith('.p'):
            layer_name = name[:-len('p') - 1]
            if layer_name in res:
                res[layer_name]['p'] = float(p.data[0])
            else:
                res[layer_name] = {'p': float(p.data[0])}
        elif name.endswith('weight'):
            if len(p.size()) == 2 or len(p.size()) == 4:
                layer_name = name[:-len('weight') - 1]
                if layer_name in res:
                    res[layer_name]['d'] = p.size()[0]
                    res[layer_name]['c'] = p.size()[1]
                else:
                    res[layer_name] = {'d': p.size()[0], 'c': p.size()[1]}
                if p.dim() > 2:
                    # (out_channels, in_channels, kernel_size[0], kernel_size[1])
                    assert p.dim() == 4
                    res[layer_name]['r'] = p.size()[2]
        else:
            continue

        res[layer_name]['m'] = m
        res[layer_name]['n'] = n
        res[layer_name]['s1'] = s1
        res[layer_name]['s2'] = s2

    for layer_name in res:
        res[layer_name] = Layer_energy(**(res[layer_name]))
    return res


def reset_Xenergy_cache(energy_info):
    for layer_name in energy_info:
        energy_info[layer_name].cached_Xenergy = None
    return energy_info

def copy_model_weights(model, W_flat, W_shapes, param_name=['weight']):
    offset = 0
    if isinstance(W_shapes, list):
        W_shapes = iter(W_shapes)
    for name, W in model.named_parameters():
        if name.strip().split(".")[-1] in param_name:
            name_, shape = next(W_shapes)
            if shape is None:
                continue
            assert name_ == name
            numel = W.numel()
            W.data.copy_(W_flat[offset: offset + numel].view(shape))
            offset += numel


def layers_nnz(model, normalized=True, param_name=['weight']):
    res = {}
    count_res = {}
    for name, W in model.named_parameters():
        if name.strip().split(".")[-1] in param_name:
            layer_name = name
            W_nz = torch.nonzero(W.data)
            if W_nz.dim() > 0:
                if not normalized:
                    res[layer_name] = W_nz.shape[0]
                else:
                    # print("{} layer nnz:{}".format(name, torch.nonzero(W.data)))
                    res[layer_name] = float(W_nz.shape[0]) / torch.numel(W)
                count_res[layer_name] = W_nz.shape[0]
            else:
                res[layer_name] = 0
                count_res[layer_name] = 0

    return res, count_res

def layers_stat(model, param_name='weight'):
    res = "########### layer stat ###########\n"
    for name, W in model.named_parameters():
        if name.endswith(param_name):
            layer_name = name[:-len(param_name) - 1]
            W_nz = torch.nonzero(W.data)
            nnz = W_nz.shape[0] / W.data.numel() if W_nz.dim() > 0 else 0.0
            W_data_abs = W.data.abs()
            res += "{:>20}".format(layer_name) + 'abs(W): min={:.4e}, mean={:.4e}, max={:.4e}, nnz={:.4f}\n'.format(W_data_abs.min().item(), W_data_abs.mean().item(), W_data_abs.max().item(), nnz)

    res += "########### layer stat ###########"
    return res

def l0proj(model, k, normalized=True, param_name=['weightA', "weightB", "weightC"]):
    # get all the weights
    W_shapes = []
    W_numel = []
    res = []
    for name, W in model.named_parameters():
        # if name.endswith(param_name):
        if name.strip().split(".")[-1] in param_name:
            if W.dim() == 1:
                W_shapes.append((name, None))
            else:
                W_shapes.append((name, W.data.shape))
                _, w_idx = torch.topk(W.data.view(-1), 1, sorted=False)
                W_numel.append((W.data.numel(), w_idx))
                res.append(W.data.view(-1))
    
    res = torch.cat(res, dim=0)
    if normalized:
        assert 0.0 <= k <= 1.0
        nnz = round(res.shape[0] * k)
    else:
        assert k >= 1 and round(k) == k
        nnz = k
    if nnz == res.shape[0]:
        z_idx = []
    else:
        _, z_idx = torch.topk(torch.abs(res), int(res.shape[0] - nnz), largest=False, sorted=False)
        offset = 0
        ttl = res.shape[0]
        WzeroInd = torch.zeros(ttl)
        WzeroInd[z_idx] = 1.0
        for item0, item1 in W_numel:
            WzeroInd[offset+item1] = 0.0
            offset += item0
        z_idx = torch.nonzero(WzeroInd)
        res[z_idx] = 0.0
        copy_model_weights(model, res, W_shapes, param_name)
    return z_idx, W_shapes

def l0proj_skip_little_matrix(model, k, normalized=True, param_name=['weightA', "weightB", "weightC"]):
    # get all the weights
    W_shapes = []
    res = []
    for name, W in model.named_parameters():
        # if name.endswith(param_name):
        if name.strip().split(".")[-1] in param_name:
            if W.dim() == 1 or W.view(-1).shape[0] < 2000:
                W_shapes.append((name, None))
            else:
                W_shapes.append((name, W.data.shape))
                res.append(W.data.view(-1))

    res = torch.cat(res, dim=0)
    if normalized:
        assert 0.0 <= k <= 1.0
        nnz = round(res.shape[0] * k)
    else:
        assert k >= 1 and round(k) == k
        nnz = k
    if nnz == res.shape[0]:
        z_idx = []
    else:
        _, z_idx = torch.topk(torch.abs(res), int(res.shape[0] - nnz), largest=False, sorted=False)
        res[z_idx] = 0.0
        copy_model_weights(model, res, W_shapes, param_name)
    return z_idx, W_shapes

def copy_model_weights_layerwise(model, W_flat, W_shapes, param_name=['weight']):
    offset = 0
    if isinstance(W_shapes, list):
        W_shapes = iter(W_shapes)
    idx = 0
    for name, W in model.named_parameters():
        if name.strip().split(".")[-1] in param_name:
            name_, shape = next(W_shapes)
            if shape is None:
                continue
            assert name_ == name
            if W_flat[idx] is not None:
                W.data.copy_(W_flat[idx].view(shape))
            idx += 1

def l0proj_layerwise(model, k, normalized=True, param_name=["weightA", "weightB", "weightC"]):
    W_shapes = []
    res = []
    z_idxes = []
    for name, W in model.named_parameters():
        if name.strip().split(".")[-1] in param_name:
            if W.dim() == 1:
                W_shapes.append((name, None))
            else:
                W_shapes.append((name, W.data.shape))
                resp = W.data.view(-1)
                if normalized:
                    assert 0.0 <= k <= 1.0
                    nnz = round(resp.shape[0] * k)
                else:
                    assert k >= 1 and round(k) == k
                    nnz = k
                
                if nnz == resp.shape[0]:
                    z_idx = []
                    z_idxes.append(z_idx)
                    res.append(None)
                else:
                    _, z_idx = torch.topk(torch.abs(resp), int(resp.shape[0] - nnz), largest=False, sorted=False)
                    resp[z_idx] = 0.0
                    # print(z_idx)
                    z_idxes.append(z_idx)
                    res.append(resp)
    copy_model_weights_layerwise(model, res, W_shapes, param_name)

    return z_idxes, W_shapes


def l0proj_varwise(model, k, normalized=True, param_name=["weightA", "weightB", "weightC"]):
    W_shapes = []
    res = []
    z_idxes = []
    for name, W in model.named_parameters():
        if name.strip().split(".")[-1] in param_name:
            if W.dim() == 1:
                W_shapes.append((name, None))
            else:
                W_shapes.append((name, W.data.shape))
                resp = W.data.view(-1)
                if normalized:
                    assert 0.0 <= k <= 1.0
                    nnz = round(resp.shape[0] * k)
                else:
                    assert k >= 1 and round(k) == k
                    nnz = k
                
                if nnz == resp.shape[0]:
                    z_idx = []
                    z_idxes.append(z_idx)
                    res.append(None)
                else:
                    _, z_idx = torch.topk(torch.abs(resp), int(resp.shape[0] - nnz), largest=False, sorted=False)
                    resp[z_idx] = 0.0
                    # print(z_idx)
                    z_idxes.append(z_idx)
                    res.append(resp)
    copy_model_weights_layerwise(model, res, W_shapes, param_name)

    return z_idxes, W_shapes

def idxproj(model, z_idx, W_shapes, param_name=['weight']):
    assert isinstance(z_idx, torch.LongTensor) or isinstance(z_idx, torch.cuda.LongTensor)
    offset = 0
    i = 0
    for name, W in model.named_parameters():
        if name.strip().split(".")[-1] in param_name:
            name_, shape = W_shapes[i]
            i += 1
            assert name_ == name
            if shape is None:
                continue
            mask = z_idx >= offset
            mask[z_idx >= (offset + W.numel())] = 0
            z_idx_sel = z_idx[mask]
            if len(z_idx_sel.shape) != 0:
                W.data.view(-1)[z_idx_sel - offset] = 0.0
            offset += W.numel()

def idxproj_layerwise(model, z_idxes, W_shapes, param_name=['weight']):
    # assert isinstance(z_idx, torch.LongTensor) or isinstance(z_idx, torch.cuda.LongTensor)
    offset = 0
    i = 0
    for name, W in model.named_parameters():
        if name.strip().split(".")[-1] in param_name:
            name_, shape = W_shapes[i]
            i += 1
            assert name_ == name
            if shape is None:
                continue
            # mask = z_idxes[offset]
            # mask[z_idx >= (offset + W.numel())] = 0
            z_idx_sel = z_idxes[offset]
            if len(z_idx_sel.shape) != 0:
                W.data.view(-1)[z_idx_sel] = 0.0
            offset += 1


def conv_cache_overlap(X_supp, padding, kernel_size, stride, k_X):
    rs = X_supp.transpose(0, 1).contiguous().view(X_supp.size(1), -1).sum(dim=1).cpu()
    rs = torch.cat([torch.zeros(padding, dtype=rs.dtype, device=rs.device),
                   rs, torch.zeros(padding, dtype=rs.dtype, device=rs.device)])
    res = 0
    beg = 0
    end = None
    while beg + kernel_size - 1 < rs.size(0):
        if end is not None:
            if beg < end:
                res += rs[beg:end].sum().item()
        n_elements = 0
        for i in range(rs.size(0) - beg):
            if n_elements + rs[beg+i] <= k_X:
                n_elements += rs[beg+i]
                if beg + i == rs.size(0) - 1:
                    end = rs.size(0)
            else:
                end = beg + i
                break
        assert end - beg >= kernel_size, 'can only hold {} rows with {} elements < {} rows in {}, cache size={}'.format(end - beg, n_elements, kernel_size, X_supp.size(), k_X)
        # print('map size={}. begin={}, end={}'.format(X_supp.size(), beg, end))
        beg += (math.floor((end - beg - kernel_size) / stride) + 1) * stride
    return res


def energy_eval(model, energy_info, e_mac=default_e_mac, e_mem=default_e_mem, e_cache=default_e_cache,
                e_rf=default_e_rf, verbose=False):
    X_nnz_dict = layers_nnz(model, normalized=False, param_name='input_mask')

    W_nnz_dict = layers_nnz(model, normalized=False, param_name='weight')

    W_energy = []
    C_energy = []
    X_energy = []
    X_supp_dict = {}
    for name, p in model.named_parameters():
        if name.endswith('input_mask'):
            layer_name = name[:-len('input_mask') - 1]
            X_supp_dict[layer_name] = (p.data != 0.0).float()

    for name, p in model.named_parameters():
        if name.endswith('weight'):
            if p is None or p.dim() == 1:
                continue
            layer_name = name[:-len('weight') - 1]
            einfo = energy_info[layer_name]

            if einfo.is_conv:
                X_nnz = einfo.h * einfo.w * einfo.c
            else:
                X_nnz = einfo.c
            if layer_name in X_nnz_dict:
                # this layer has sparse input
                X_nnz = X_nnz_dict[layer_name]

            if layer_name in X_supp_dict:
                X_supp = X_supp_dict[layer_name].unsqueeze(0)
            else:
                if einfo.is_conv:
                    X_supp = torch.ones(1, p.size(1), int(energy_info[layer_name].h),
                                        int(energy_info[layer_name].w), dtype=p.dtype, device=p.device)
                else:
                    X_supp = None

            unfoldedX = None

            # input data access energy
            if einfo.is_conv:
                h_, w_ = max(0.0, math.floor((einfo.h + 2 * einfo.p - einfo.r) / einfo.xi) + 1), max(0.0, math.floor((einfo.w + 2 * einfo.p - einfo.r) / einfo.xi) + 1)
                unfoldedX = F.unfold(X_supp, kernel_size=int(einfo.r), padding=int(einfo.p), stride=int(einfo.xi)).squeeze(0)
                assert unfoldedX.size(1) == h_ * w_, 'unfolded X size={}, but h_ * w_ = {}, W.size={}'.format(unfoldedX.size(), h_ * w_, p.size())
                unfoldedX_nnz = (unfoldedX != 0.0).float().sum().item()

                X_energy_cache = unfoldedX_nnz * math.ceil((float(einfo.d) / einfo.g) / einfo.n) * e_cache
                X_energy_rf = unfoldedX_nnz * math.ceil(float(einfo.d) / einfo.g) * e_rf

                X_energy_mem = X_nnz * e_mem + \
                               conv_cache_overlap(X_supp.squeeze(0), int(einfo.p), int(einfo.r), int(einfo.xi), default_s1) * e_mem + \
                               unfoldedX.size(1) * einfo.d * e_mem
                X_energy_this = X_energy_mem + X_energy_rf + X_energy_cache
            else:
                X_energy_cache = math.ceil(float(einfo.d) / einfo.n) * e_cache * X_nnz
                X_energy_rf = float(einfo.d) * e_rf * X_nnz
                X_energy_mem = e_mem * (math.ceil(float(einfo.d) / einfo.n) * max(0.0, X_nnz - default_s1)
                                        + min(X_nnz, default_s1)) + e_mem * float(einfo.d)

                X_energy_this = X_energy_mem + X_energy_rf + X_energy_cache

            einfo.cached_Xenergy = X_energy_this
            X_energy.append(X_energy_this)

            # kernel weights data access energy
            if einfo.is_conv:
                output_hw = unfoldedX.size(1)
                W_energy_cache = math.ceil(output_hw / einfo.m) * W_nnz_dict[layer_name] * e_cache
                W_energy_rf = output_hw * W_nnz_dict[layer_name] * e_rf
                W_energy_mem = (math.ceil(output_hw / einfo.m) * max(0.0, W_nnz_dict[layer_name] - default_s2)\
                               + min(default_s2, W_nnz_dict[layer_name])) * e_mem
                W_energy_this = W_energy_cache + W_energy_rf + W_energy_mem
            else:
                W_energy_this = einfo.get_beta(e_mem, e_cache, e_rf, in_cache=None) * W_nnz_dict[layer_name]
            W_energy.append(W_energy_this)

            # computation enregy
            if einfo.is_conv:
                N_mac = torch.sum(
                    F.conv2d(X_supp, (p.data != 0.0).float(), None, int(energy_info[layer_name].xi),
                             int(energy_info[layer_name].p), 1, int(energy_info[layer_name].g))).item()
                C_energy_this = e_mac * N_mac
            else:
                C_energy_this = e_mac * (W_nnz_dict[layer_name])

            C_energy.append(C_energy_this)

            if verbose:
                print("Layer: {}, W_energy={:.2e}, C_energy={:.2e}, X_energy={:.2e}".format(layer_name,
                                                                                            W_energy[-1],
                                                                                            C_energy[-1],
                                                                                            X_energy[-1]))

    return {'W': sum(W_energy), 'C': sum(C_energy), 'X': sum(X_energy)}


def energy_eval_relax(model, energy_info, e_mac=default_e_mac, e_mem=default_e_mem, e_cache=default_e_cache,
                e_rf=default_e_rf, verbose=False):
    W_nnz_dict = layers_nnz(model, normalized=False, param_name='weight')

    W_energy = []
    C_energy = []
    X_energy = []
    X_supp_dict = {}
    for name, p in model.named_parameters():
        if name.endswith('input_mask'):
            layer_name = name[:-len('input_mask') - 1]
            X_supp_dict[layer_name] = (p.data != 0.0).float()

    for name, p in model.named_parameters():
        if name.endswith('weight'):
            if p is None or p.dim() == 1:
                continue
            layer_name = name[:-len('weight') - 1]
            assert energy_info[layer_name].cached_Xenergy is not None
            X_energy.append(energy_info[layer_name].cached_Xenergy)
            assert X_energy[-1] > 0
            if not energy_info[layer_name].is_conv:
                # in_cache is not needed in fc layers
                in_cache = None
                W_energy.append(
                    energy_info[layer_name].get_beta(e_mem, e_cache, e_rf, in_cache) * W_nnz_dict[layer_name])
                C_energy.append(e_mac * (W_nnz_dict[layer_name]))
                if verbose:
                    knapsack_weight1 = energy_info[layer_name].get_knapsack_weight_W(e_mac, e_mem, e_cache, e_rf,
                                                                                     in_cache=None, crelax=True)
                    if hasattr(knapsack_weight1, 'mean'):
                        knapsack_weight1 = knapsack_weight1.mean()
                    print(layer_name + " weight: {:.4e}".format(knapsack_weight1))

            else:
                beta1 = energy_info[layer_name].get_beta(e_mem, e_cache, e_rf, in_cache=True)
                beta2 = energy_info[layer_name].get_beta(e_mem, e_cache, e_rf, in_cache=False)

                W_nnz = W_nnz_dict[layer_name]
                W_energy_this = beta1 * min(energy_info[layer_name].s2, W_nnz) + beta2 * max(0, W_nnz - energy_info[
                    layer_name].s2)
                W_energy.append(W_energy_this)
                C_energy.append(e_mac * energy_info[layer_name].h_ * float(energy_info[layer_name].w_) * W_nnz)

            if verbose:
                print("Layer: {}, W_energy={:.2e}, C_energy={:.2e}, X_energy={:.2e}".format(layer_name,
                                                                                            W_energy[-1],
                                                                                            C_energy[-1],
                                                                                            X_energy[-1]))

    return {'W': sum(W_energy), 'C': sum(C_energy), 'X': sum(X_energy)}


def energy_proj(model, energy_info, budget, e_mac=default_e_mac, e_mem=default_e_mem, e_cache=default_e_cache,
                e_rf=default_e_rf, grad=False, in_place=True, preserve=0.0, param_name='weight'):
    knapsack_bound = budget
    param_flats = []
    knapsack_weight_all = []
    score_all = []
    param_shapes = []
    bound_bias = 0.0

    for name, p in model.named_parameters():
        if name.endswith(param_name):
            if p is None or (param_name == 'weight' and p.dim() == 1):
                # skip batch_norm layer
                param_shapes.append((name, None))
                continue
            else:
                param_shapes.append((name, p.data.shape))

            layer_name = name[:-len(param_name) - 1]
            assert energy_info[layer_name].cached_Xenergy is not None
            if grad:
                p_flat = p.grad.data.view(-1)
            else:
                p_flat = p.data.view(-1)
            score = p_flat ** 2

            if param_name == 'weight':
                knapsack_weight = energy_info[layer_name].get_knapsack_weight_W(e_mac, e_mem, e_cache, e_rf,
                                                                                in_cache=True, crelax=True)
                if hasattr(knapsack_weight, 'view'):
                    knapsack_weight = knapsack_weight.view(1, -1, 1, 1)
                knapsack_weight = torch.zeros_like(p.data).add_(knapsack_weight).view(-1)

                # preserve part of weights
                if preserve > 0.0:
                    if preserve > 1:
                        n_preserve = preserve
                    else:
                        n_preserve = round(p_flat.numel() * preserve)
                    _, preserve_idx = torch.topk(score, k=n_preserve, largest=True, sorted=False)
                    score[preserve_idx] = float('inf')

                if energy_info[layer_name].is_conv and p_flat.numel() > energy_info[layer_name].s2:
                    delta = energy_info[layer_name].get_beta(e_mem, e_cache, e_rf, in_cache=False) \
                            - energy_info[layer_name].get_beta(e_mem, e_cache, e_rf, in_cache=True)
                    assert delta >= 0
                    _, out_cache_idx = torch.topk(score, k=p_flat.numel() - energy_info[layer_name].s2, largest=False,
                                                  sorted=False)
                    knapsack_weight[out_cache_idx] += delta

                bound_const = energy_info[layer_name].cached_Xenergy

                assert bound_const > 0
                bound_bias += bound_const
                knapsack_bound -= bound_const

            else:
                raise ValueError('not supported parameter name')

            score_all.append(score)
            knapsack_weight_all.append(knapsack_weight)
            # print(layer_name, X_nnz, knapsack_weight)
            param_flats.append(p_flat)

    param_flats = torch.cat(param_flats, dim=0)
    knapsack_weight_all = torch.cat(knapsack_weight_all, dim=0)
    score_all = torch.cat(score_all, dim=0) / knapsack_weight_all

    _, sorted_idx = torch.sort(score_all, descending=True)
    cumsum = torch.cumsum(knapsack_weight_all[sorted_idx], dim=0)
    res_nnz = torch.nonzero(cumsum <= knapsack_bound).max()
    z_idx = sorted_idx[-(param_flats.numel() - res_nnz):]

    if in_place:
        param_flats[z_idx] = 0.0
        copy_model_weights(model, param_flats, param_shapes, param_name)
    return z_idx, param_shapes

# energy_info = build_energy_info(model)
# energy_estimator = lambda m: sum(energy_eval(m, energy_info, verbose=False).values())

class myConv2d(nn.Conv2d):
    def __init__(self, h_in, w_in, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(myConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)
        self.h_in = h_in
        self.w_in = w_in
        self.xi = Parameter(torch.LongTensor(1), requires_grad=False)
        self.xi.data[0] = stride
        self.g = Parameter(torch.LongTensor(1), requires_grad=False)
        self.g.data[0] = groups
        self.p = Parameter(torch.LongTensor(1), requires_grad=False)
        self.p.data[0] = padding

    def __repr__(self):
        s = ('{name}({h_in}, {w_in}, {in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class FixHWConv2d(myConv2d):
    def __init__(self, h_in, w_in, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(FixHWConv2d, self).__init__(h_in, w_in, in_channels, out_channels, kernel_size, stride,
                                          padding, dilation, groups, bias)

        self.hw = Parameter(torch.LongTensor(2), requires_grad=False)
        self.hw.data[0] = h_in
        self.hw.data[1] = w_in

    def forward(self, input):
        # Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        assert input.size(2) == self.hw.data[0] and input.size(3) == self.hw.data[1], 'input_size={}, but hw={}'.format(
            input.size(), self.hw.data)
        return super(FixHWConv2d, self).forward(input)


class SparseConv2d(myConv2d):
    def __init__(self, h_in, w_in, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(SparseConv2d, self).__init__(h_in, w_in, in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias)

        self.input_mask = Parameter(torch.Tensor(in_channels, h_in, w_in))
        self.input_mask.data.fill_(1.0)

    def forward(self, input):
        # print("###{}, {}".format(input.size(), self.input_mask.size()))
        return super(SparseConv2d, self).forward(input * self.input_mask)


def conv2d_out_dim(dim, kernel_size, padding=0, stride=1, dilation=1, ceil_mode=False):
    if ceil_mode:
        return int(math.ceil((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
    else:
        return int(math.floor((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))


class MyLeNet5(nn.Module):
    def __init__(self, conv_class=FixHWConv2d):
        super(MyLeNet5, self).__init__()
        h = 32
        w = 32
        feature_layers = []
        # conv
        feature_layers.append(conv_class(h, w, 1, 6, kernel_size=5))
        h = conv2d_out_dim(h, kernel_size=5)
        w = conv2d_out_dim(w, kernel_size=5)
        feature_layers.append(nn.ReLU(inplace=True))
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        h = conv2d_out_dim(h, kernel_size=2, stride=2)
        w = conv2d_out_dim(w, kernel_size=2, stride=2)
        # conv
        feature_layers.append(conv_class(h, w, 6, 16, kernel_size=5))
        h = conv2d_out_dim(h, kernel_size=5)
        w = conv2d_out_dim(w, kernel_size=5)
        feature_layers.append(nn.ReLU(inplace=True))
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        h = conv2d_out_dim(h, kernel_size=2, stride=2)
        w = conv2d_out_dim(w, kernel_size=2, stride=2)

        self.features = nn.Sequential(*feature_layers)

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 16 * 5 * 5)
        x = self.classifier(x)
        return x



class MyCaffeLeNet(nn.Module):
    def __init__(self, conv_class=FixHWConv2d):
        super(MyCaffeLeNet, self).__init__()
        h = 28
        w = 28
        feature_layers = []
        # conv
        feature_layers.append(conv_class(h, w, 1, 20, kernel_size=5))
        h = conv2d_out_dim(h, kernel_size=5)
        w = conv2d_out_dim(w, kernel_size=5)
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        h = conv2d_out_dim(h, kernel_size=2, stride=2)
        w = conv2d_out_dim(w, kernel_size=2, stride=2)
        # conv
        feature_layers.append(conv_class(h, w, 20, 50, kernel_size=5))
        h = conv2d_out_dim(h, kernel_size=5)
        w = conv2d_out_dim(w, kernel_size=5)
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        h = conv2d_out_dim(h, kernel_size=2, stride=2)
        w = conv2d_out_dim(w, kernel_size=2, stride=2)

        self.features = nn.Sequential(*feature_layers)

        self.classifier = nn.Sequential(
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 50 * 4 * 4)
        x = self.classifier(x)
        return x

class SapUnit(nn.Module):
    def __init__(self, r):
        super(SapUnit, self).__init__()
        self.r = r

    def forward(self, x):

        if self.training:
            return x
        else:

            # print(x.shape)
            hidden = x.view(x.size(0), -1)
            # print(hidden.shape)
            # print(hidden)
            hidden_abs = hidden.data.abs()
            # print(hidden.shape)
            rate = int(self.r * hidden_abs.size(1))
            # print(rate)
            # print("hidden_abs")
            # print(hidden_abs.sort(descending=True)[0])
            # nnsoftabs = F.softmax(hidden_abs, dim=1)
            with torch.no_grad():
                # hidden_abs.clamp_(min=1e-1, max=5)
                hidden_p = hidden_abs / (torch.sum(hidden_abs, dim=1, keepdim=True) + 1e-10)
                # hidden_p += 1e-10
                hidden_idx = torch.multinomial(hidden_p, rate, replacement=True)
            # print(hidden_idx.shape)
            
            # print("hidden_p")
            # print(hidden_p.sort(descending=True)[0])
            # print(hidden_p.shape)
            # hidden_out = hidden.data.clone()
            # print(torch.arange(hidden_idx.size(0)).long())
            # print(hidden_idx)
            # print(hidden[torch.arange(hidden_idx.size(0)).long().unsqueeze(1), hidden_idx].shape)
            first_d = torch.arange(hidden_idx.size(0)).long().unsqueeze(1)
            hidden_out = torch.zeros_like(hidden)
            pre_compute = 1 / (1 - torch.exp(rate * torch.log(1 - hidden_p[first_d, hidden_idx])) )
            inf_idx = torch.isinf(pre_compute)
            pre_compute[inf_idx] = 1/(rate * hidden_p[inf_idx])
            hidden_out[first_d, hidden_idx] = hidden[first_d, hidden_idx] * pre_compute #* (1/(1-(1-hidden_p[first_d, hidden_idx])**(rate))) #+ hidden[first_d, hidden_idx]
            # hidden_out = hidden * (1/(1-(1-hidden_p).pow(rate)))
            # print((1/(1-(1-hidden_p[first_d, hidden_idx])**(rate))).max().item())
            # print(((1- hidden_p[first_d, hidden_idx]) ** rate).max().item())
            # hidden_out[, hidden_idx] = \
                # hidden[torch.arange(hidden_idx.size(0)).long().unsqueeze(1), hidden_idx] * (1 / (1 - (1 - hidden_p[torch.arange(hidden_idx.size(0)).long().unsqueeze(1), hidden_idx]).pow(rate)) + 1)
            # hidden_out = hidden * (1/(1-(1-hidden_p).pow(rate)) + 1)
            # print("hidden_out")
            # print(hidden_out.sort(descending=True)[0])
            # hidden_out = hidden_out - hidden

            # hidden_out = hidden_out.reshape(x.shape)
            
            # print('in sum {}'.format(hidden.sum().item()))
            # print('prob: {}')
            # print(hidden_p[first_d, hidden_idx].view(-1).sort()[0])
            # assert hidden_p[first_d, hidden_idx].sum() != float('inf')
            # temp = hidden_p[first_d, hidden_idx]#.view(-1).min().item()
            # # print('temp={}'.format(temp))
            # idx = torch.isinf(1/(1-torch.exp(rate * torch.log(1-temp))))
            # print('inf -- prob:')
            # print(temp[idx])
            # temp[idx] = 1/(rate * hidden_p[idx])
            # temp2 = temp[idx]
            # # if len(temp2) != 0:
            # #     print('1/(1-(1-{}) ** {} ={}'.format(temp2.item(), rate, 1/(1-(1-temp2.item())**rate)))
            # #     print((1/(1-(1-temp)**(rate))).sum())
            # print('out sum {}'.format(hidden_out.sum().item()))
            # if hidden_out.sum().item() == float('inf'):
            #     exit()

            hidden_out = hidden_out.view(*x.shape)
            return hidden_out

# def sapunit(x, r):
#     hidden = x.view(x.size(0), -1)
#     hidden_abs = hidden.abs()
#     rate = int(r * hidden_abs.size(0))
#     hidden_idx = torch.multinomial(hidden_abs, rate, replacement=True)
#     hidden_p = hidden_abs / torch.sum(hidden_abs, dim=1, keepdim=True)
#     hidden_out = hidden.copy()
#     hidden_out[torch.arange(hidden_idx.size(0)).long()][hidden_idx] = \
#         hidden[torch.arange(hidden_idx.size(0)).long()][hidden_idx] (1 / (1 - (1 - hidden_p[torch.arange(hidden_idx.size(0)).long()][hidden_idx]).pow(rate)) + 1)
#     hidden_out = hidden_out - hidden
    
#     return hidden_out

class MySapCaffeLeNet(nn.Module):
    def __init__(self, conv_class=FixHWConv2d, r=1.0):
        super(MySapCaffeLeNet, self).__init__()
        h = 28
        w = 28
        feature_layers = []
        # conv
        feature_layers.append(conv_class(h, w, 1, 20, kernel_size=5))
        h = conv2d_out_dim(h, kernel_size=5)
        w = conv2d_out_dim(w, kernel_size=5)
        # feature_layers.append(SapUnit(r=r))
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        h = conv2d_out_dim(h, kernel_size=2, stride=2)
        w = conv2d_out_dim(w, kernel_size=2, stride=2)
        feature_layers.append(SapUnit(r=r))
        # conv
        feature_layers.append(conv_class(h, w, 20, 50, kernel_size=5))
        h = conv2d_out_dim(h, kernel_size=5)
        w = conv2d_out_dim(w, kernel_size=5)
        # feature_layers.append(SapUnit(r=r))
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        h = conv2d_out_dim(h, kernel_size=2, stride=2)
        w = conv2d_out_dim(w, kernel_size=2, stride=2)
        feature_layers.append(SapUnit(r=r))

        self.features = nn.Sequential(*feature_layers)

        self.classifier = nn.Sequential(
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(inplace=True),
            SapUnit(r=r),
            nn.Linear(500, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 50 * 4 * 4)
        x = self.classifier(x)
        return x

def masked_layers_hw(cfg, c_in, h_in, w_in, use_mask=True, batch_norm=False):
    afun = nn.ReLU()
    layers = []
    for i, v in enumerate(cfg):
        if v == 'M':
            Mstride = 2
            Mkernel = 2
            Mpadding = 0
            h_out = conv2d_out_dim(h_in, padding=Mpadding, kernel_size=Mkernel, stride=Mstride)
            w_out = conv2d_out_dim(w_in, padding=Mpadding, kernel_size=Mkernel, stride=Mstride)

            layers += [nn.MaxPool2d(kernel_size=Mkernel, stride=Mstride)]
        else:
            Cpadding = v[1] if isinstance(v, tuple) else 1
            c_out = v[0] if isinstance(v, tuple) else v
            Ckernel = 3
            h_out = conv2d_out_dim(h_in, padding=Cpadding, kernel_size=Ckernel)
            w_out = conv2d_out_dim(w_in, padding=Cpadding, kernel_size=Ckernel)

            if use_mask or i == 0:
                conv2d = SparseConv2d(h_in, w_in, c_in, c_out, kernel_size=Ckernel, padding=Cpadding)
            else:
                conv2d = FixHWConv2d(h_in, w_in, c_in, c_out, kernel_size=Ckernel, padding=Cpadding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c_out, affine=False), afun]
            else:
                layers += [conv2d, afun]
            c_in = c_out

        h_in = h_out
        w_in = w_out
    return nn.Sequential(*layers)


class CaffeLeNet(nn.Module):
    def __init__(self):
        super(CaffeLeNet, self).__init__()
        feature_layers = []
        # conv
        # feature_layers.append(conv_class(h, w, 1, 20, kernel_size=5))
        feature_layers.append(nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5))
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # conv
        # feature_layers.append(conv_class(h, w, 20, 50, kernel_size=5))
        feature_layers.append(nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5))
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*feature_layers)

        self.classifier = nn.Sequential(
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 50 * 4 * 4)
        x = self.classifier(x)
        return x

    def param_modules(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                yield module

    def named_param_modules(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                yield name, module

    def get_ranks(self):
        ranks = []
        for m in self.param_modules():
            # print(type(m))
            if isinstance(m, nn.Conv2d):
                if m.groups != 1:
                    continue
                c, d = m.in_channels, m.out_channels
                rank = min(c * m.kernel_size[0] * m.kernel_size[1], d)
            else:
                c, d = m.in_features, m.out_features
                rank = min(c, d)
            ranks.append(rank)
        return ranks

    @staticmethod
    def scale_tosame(u, v, w):
        print("dist0 {}".format(torch.norm(v.matmul(u) - w)))
        uscale = torch.sqrt(torch.mean((u * u).view(-1)))
        vscale = torch.sqrt(torch.mean((v * v).view(-1)))
        wscale = torch.sqrt(torch.mean((w * w).view(-1)))
        print("scale pre, U {}, V{}".format(uscale, vscale))
        u = u / uscale * torch.sqrt(uscale * vscale)
        v = v / vscale * torch.sqrt(vscale * uscale)
        # return u, v, torch.zeros_like(w)
        print("scale now, U {}, V{}".format(torch.sqrt(torch.mean((u * u).view(-1))), torch.sqrt(torch.mean((v * v).view(-1)))))
        print("dist1 {}".format(torch.norm(v.matmul(u) - w)))
        t = (5 ** (0.5) - 1) / (2.)
        alpha = (1-t) ** 0.5
        print("t {}, alpha {}".format(t, alpha))
        print("dist2 {}".format(torch.norm(v.matmul(u) * alpha * alpha - w * (1-t))))
        print("scale now, U {}, V {}, C {}".format(
            torch.sqrt(torch.mean((u * u * alpha **4).view(-1))), 
            torch.sqrt(torch.mean((v * v * alpha **4).view(-1))),
            torch.sqrt(torch.mean((w * w * t **2).view(-1)))
        ))
        return u * alpha, v * alpha, w * t

    def raw_weights(self, ranks):
        weights_list=[]
        i = 0
        for m in self.param_modules():
            if isinstance(m, nn.Conv2d) and m.groups != 1:
                continue
            elif isinstance(m, nn.Conv2d):
                weight = m.weight.view(m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1]).data
                if ranks[i] == m.in_channels * m.kernel_size[0] * m.kernel_size[1]:
                    weightA = None
                    weightB = weight
                    weightC = None
                else:
                    weightA = weight
                    weightB = None
                    weightC = None
            else:
                weight = m.weight.data
                if ranks[i] == m.in_features:
                    weightA = None
                    weightB = weight
                    weightC = None
                else:
                    weightA = weight
                    weightB = None
                    weightC = None
        
            i+= 1
            
            weights_list.append((weightA, weightB, weightC))
        return weights_list

    def svd_weights(self, ranks=None):
        weights_list = []
        i = 0
        for m in self.param_modules():
            if isinstance(m, nn.Conv2d) and m.groups != 1:
                continue
            if isinstance(m, nn.Conv2d):
                weight = m.weight.view(m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1]).data
                U, S, V = torch.svd(weight)
            else:
                weight = m.weight.data
                U, S, V = torch.svd(weight)
            
            if ranks is None:
                S_sqrt = torch.sqrt(S)
                weight_B = U * S_sqrt
                weight_A = S_sqrt.view(-1, 1) * V.t()
                # weights_list.append((weight_A, weight_B))
            else:
                S_sqrt = torch.sqrt(S[:ranks[i]])
                weight_B = U[:, :ranks[i]] * S_sqrt
                weight_A = S_sqrt.view(-1, 1) * V.t()[:ranks[i], :]
            
                # weights_list.append((weightA, weightB))
            weightA, weightB, weightC = self.scale_tosame(weight_A, weight_B, weight)
            # restore_w = weight_B.matmul(weight_A) #+ weightC
            restore_w = weightB.matmul(weightA) + weightC

            print("dist {}".format(torch.norm(restore_w - weight)))
            weights_list.append((weightA, weightB, weightC))
            # print("A {}, B {}, C {}".format(weightA, weightB, weightC))
            i += 1
        return weights_list

    def svd_weights_v2(self, ranks=None):
        weights_list = []
        i = 0
        for m in self.param_modules():
            if isinstance(m, nn.Conv2d) and m.groups != 1:
                continue
            if isinstance(m, nn.Conv2d):
                weight = m.weight.view(m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1]).data
            else:
                weight = m.weight.data
            
            # t = (math.sqrt(5) - 1)/2
            t = 0.0
            if torch.cuda.is_available():
                C = t * weight * (torch.cuda.FloatTensor(weight.size()).uniform_() > 0.5).type(torch.FloatTensor)
            else:
                C = t * weight * (torch.FloatTensor(weight.size()).uniform_() > 0.5).type(torch.FloatTensor)
            
            U, S, V = torch.svd(weight - C)
            
            if ranks is None:
                S_sqrt = torch.sqrt(S)
                weight_B = U * S_sqrt
                weight_A = S_sqrt.view(-1, 1) * V.t()
            else:
                S_sqrt = torch.sqrt(S[:ranks[i]])
                weight_B = U[:, :ranks[i]] * S_sqrt
                weight_A = S_sqrt.view(-1, 1) * V.t()[:ranks[i], :]
            
                # weights_list.append((weightA, weightB))
            u, v = weight_A, weight_B
            uscale = torch.sqrt(torch.mean((u * u).view(-1)))
            vscale = torch.sqrt(torch.mean((v * v).view(-1)))
            u = u / uscale * torch.sqrt(uscale * vscale)
            v = v / vscale * torch.sqrt(vscale * uscale)
            weightA, weightB, weightC = u, v, C
            # restore_w = weight_B.matmul(weight_A) #+ weightC
            restore_w = weightB.matmul(weightA) + weightC

            print("dist {}".format(torch.norm(restore_w - weight)))
            weights_list.append((weightA, weightB, weightC))
            # print("A {}, B {}, C {}".format(weightA, weightB, weightC))
            i += 1
        return weights_list


    def svd_weights_v3(self, ranks):
        weights_list = []
        i = 0
        for m in self.param_modules():
            if isinstance(m, nn.Conv2d) and m.groups != 1:
                continue
            if isinstance(m, nn.Conv2d):
                weight = m.weight.view(m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1]).data
            else:
                weight = m.weight.data
            
            # t = (math.sqrt(5) - 1)/2
            t = 0.0
            if torch.cuda.is_available():
                C = t * weight * (torch.cuda.FloatTensor(weight.size()).uniform_() > 0.5).type(torch.FloatTensor)
            else:
                C = t * weight * (torch.FloatTensor(weight.size()).uniform_() > 0.5).type(torch.FloatTensor)
            
            U, S, V = torch.svd(weight - C)
            
            S_sqrt = torch.sqrt(S)
            weight_B = U * S_sqrt
            weight_A = S_sqrt.view(-1, 1) * V.t()
            
            if torch.cuda.is_available():
                eye_term = torch.eye(ranks[i])
            else:
                eye_term = torch.eye(ranks[i])

            u, v = weight_A, weight_B
            
            uscale = torch.sqrt(torch.mean((u * u).view(-1)))
            vscale = torch.sqrt(torch.mean((v * v).view(-1)))
            u = u / uscale * torch.sqrt(uscale * vscale)
            v = v / vscale * torch.sqrt(vscale * uscale)
            if (isinstance(m, nn.Conv2d) and ranks[i] == m.in_channels * m.kernel_size[0] * m.kernel_size[1]) or (isinstance(m, nn.Linear) and ranks[i] == m.in_features):
                u = u - eye_term
                restore_w = v.matmul(u + eye_term) + C
            else:
                v = v - eye_term
                restore_w = (v+eye_term).matmul(u) + C
            weightA, weightB, weightC = u, v, C
            # restore_w = weight_B.matmul(weight_A) #+ weightC

            print("dist {}".format(torch.norm(restore_w - weight)))
            weights_list.append((weightA, weightB, weightC))
            # print("A {}, B {}, C {}".format(weightA, weightB, weightC))
            i += 1
        return weights_list

    def svd_lowrank_weights(self, ranks=None):
        weights_list = []
        i = 0
        for m in self.param_modules():
            if isinstance(m, nn.Conv2d) and m.groups != 1:
                continue
            if isinstance(m, nn.Conv2d):
                weight = m.weight.view(m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1]).data
            else:
                weight = m.weight.data
            
            U, S, V = torch.svd(weight)
            
            if ranks is None:
                S_sqrt = torch.sqrt(S)
                weight_B = U * S_sqrt
                weight_A = S_sqrt.view(-1, 1) * V.t()
            else:
                S_sqrt = torch.sqrt(S[:ranks[i]])
                weight_B = U[:, :ranks[i]] * S_sqrt
                weight_A = S_sqrt.view(-1, 1) * V.t()[:ranks[i], :]
            
            # restore_w = weight_B.matmul(weight_A) #+ weightC
            restore_w = weight_B.matmul(weight_A)

            print("dist {}".format(torch.norm(restore_w - weight)))
            weights_list.append((weight_A, weight_B))
            # print("A {}, B {}, C {}".format(weightA, weightB, weightC))
            i += 1
        return weights_list

    def svd_global_lowrank_weights(self, k):
        weights_list = []
        i = 0
        res = []
        resU = []
        resV = []
        Sshapes = []
        orig_weights = []
        
        for name, m in self.named_param_modules():
            if isinstance(m, nn.Conv2d) and m.groups != 1:
                continue
            if isinstance(m, nn.Conv2d):
                weight = m.weight.view(m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1]).data
            else:
                weight = m.weight.data
            orig_weights.append(weight)
            U, S, V = torch.svd(weight)
            Sshapes.append(S.shape)

            res.append(S)
            resU.append(U)
            resV.append(V)
        
        res1 = torch.cat(res, dim=0)
        _, z_idx = torch.topk(torch.abs(res1), int(res1.shape[0] * (1-k)), largest=False, sorted=False)
        res1[z_idx] = 0.0
        offset = 0
        ranks = []
        for i in range(len(res)):
            S, U, V = res[i], resU[i], resV[i]
            numel = S.numel()
            nnz_idx = torch.nonzero(res1[offset: offset+numel])
            rank = nnz_idx.shape[0]
            print(rank)
            if rank ==0:
                rank = 1
            ranks.append(rank)
            S_sqrt = torch.sqrt(S[:rank])
            weight_B = U[:, :rank] * S_sqrt
            weight_A = S_sqrt.view(-1, 1) * V.t()[:rank, :]
            print("{} {} {}".format(S_sqrt.shape, weight_B.shape, weight_A.shape))
            # restore_w = weight_B.matmul(weight_A) #+ weightC
            restore_w = weight_B.matmul(weight_A)

            print("dist {}".format(torch.norm(restore_w - orig_weights[i])))
            weights_list.append((weight_A, weight_B))
            # print("A {}, B {}, C {}".format(weightA, weightB, weightC))
            offset += numel
        return weights_list, ranks


    def lu_weights(self):
        weights_list = []
        for m in self.param_modules():
            if isinstance(m, nn.Conv2d) and m.groups != 1:
                continue
            if isinstance(m, nn.Conv2d):
                mdata = m.weight.view(m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1]).data.cpu().numpy()
            else:
                mdata = m.weight.data.cpu().numpy()

            L, U = sl.lu(mdata, permute_l=True)
            L = torch.Tensor(L)
            U = torch.Tensor(U)
            W = torch.Tensor(mdata)
            weightA, weightB, weightC = self.scale_tosame(U, L, W)
            weights_list.append((weightA, weightB, weightC))

        return weights_list

    def lu_weights_v2(self):
        weights_list = []
        i = 0

        for m in self.param_modules():
            if isinstance(m, nn.Conv2d) and m.groups != 1:
                continue
            if isinstance(m, nn.Conv2d):
                weight = m.weight.view(m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1]).data
            else:
                weight = m.weight.data
            
            # t = (math.sqrt(5) - 1)/2
            t = 1e-3
            if torch.cuda.is_available():
                C = t * weight * (torch.cuda.FloatTensor(weight.size()).uniform_() > 0.9).type(torch.FloatTensor)
            else:
                C = t * weight * (torch.FloatTensor(weight.size()).uniform_() > 0.9).type(torch.FloatTensor)

            mdata = (weight - C).cpu().numpy()
            
            L, U = sl.lu(mdata, permute_l=True)
            
            weight_A = torch.Tensor(U)
            weight_B = torch.Tensor(L)
            u, v = weight_A, weight_B
            uscale = torch.sqrt(torch.mean((u * u).view(-1)))
            vscale = torch.sqrt(torch.mean((v * v).view(-1)))
            u = u / uscale * torch.sqrt(uscale * vscale)
            v = v / vscale * torch.sqrt(vscale * uscale)
            print("scale u {}, v {}".format(torch.sqrt(torch.mean((u * u).view(-1))), torch.sqrt(torch.mean((v * v).view(-1)))))
            weightA, weightB, weightC = u, v, C
            print("size A {}, B {}, C {}".format(weightA.shape, weightB.shape, weightC.shape))
            restore_w = weightB.matmul(weightA) + weightC

            print("dist {}".format(torch.norm(restore_w - weight)))
            weights_list.append((weightA, weightB, weightC))
            # print("A {}, B {}, C {}".format(weightA, weightB, weightC))
            i += 1
        return weights_list

    def lu_weights_v3(self, ranks):
        weights_list = []
        i = 0

        for m in self.param_modules():
            if isinstance(m, nn.Conv2d) and m.groups != 1:
                continue
            if isinstance(m, nn.Conv2d):
                weight = m.weight.view(m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1]).data
            else:
                weight = m.weight.data
            
            # t = (math.sqrt(5) - 1)/2
            t = 0.0
            if torch.cuda.is_available():
                C = t * weight * (torch.cuda.FloatTensor(weight.size()).uniform_() > 0.9).type(torch.FloatTensor)
            else:
                C = t * weight * (torch.FloatTensor(weight.size()).uniform_() > 0.9).type(torch.FloatTensor)

            mdata = (weight - C).cpu().numpy()
            
            L, U = sl.lu(mdata, permute_l=True)
            
            weight_A = torch.Tensor(U)
            weight_B = torch.Tensor(L)
            u, v = weight_A, weight_B
            uscale = torch.sqrt(torch.mean((u * u).view(-1)))
            vscale = torch.sqrt(torch.mean((v * v).view(-1)))
            u = u / uscale * torch.sqrt(uscale * vscale)
            v = v / vscale * torch.sqrt(vscale * uscale)
            print("scale u {}, v {}".format(torch.sqrt(torch.mean((u * u).view(-1))), torch.sqrt(torch.mean((v * v).view(-1)))))
            eye_term = torch.eye(ranks[i])
            if (isinstance(m, nn.Conv2d) and ranks[i] == m.in_channels * m.kernel_size[0] * m.kernel_size[1]) or (isinstance(m, nn.Linear) and ranks[i] == m.in_features):
                u = u - eye_term
                restore_w = v.matmul(u + eye_term) + C
            else:
                v = v - eye_term
                restore_w = (v+eye_term).matmul(u) + C
            weightA, weightB, weightC = u, v, C
            print("size A {}, B {}, C {}".format(weightA.shape, weightB.shape, weightC.shape))
            restore_w = weightB.matmul(weightA) + weightC

            print("dist {}".format(torch.norm(restore_w - weight)))
            weights_list.append((weightA, weightB, weightC))
            # print("A {}, B {}, C {}".format(weightA, weightB, weightC))
            i += 1
        return weights_list

#######################################################################
#########################-Factorize Layer Design-######################
#######################################################################
            
class Linearsp(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super(Linearsp, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        print("rank {}, in_features {}, out_features {}".format(rank, in_features, out_features))
        assert rank <= min(in_features, out_features)
        self.rank = rank
        self.weightA = Parameter(torch.Tensor(rank, in_features))
        self.weightB = Parameter(torch.Tensor(out_features, rank))
        self.weightC = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weightA.size(1))
        self.weightA.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.weightB.size(1))
        self.weightB.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.weightC.size(1))
        self.weightC.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # print("B {}, A{}, C{}".format(self.weightB.size(), self.weightA.size(), self.weightC.size()))
        weight = self.weightB.matmul(self.weightA) + self.weightC
        return F.linear(input, weight, self.bias)
        # return F.linear(input.matmul(self.weightA.t()), self.weightB, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Conv2dsp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rank, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dsp, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        assert groups == 1, 'does not support grouped convolution yet'
        assert rank <= min(in_channels * kernel_size[0] * kernel_size[1], out_channels)
        self.rank = rank
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weightA = Parameter(torch.Tensor(rank, in_channels, *kernel_size))
        self.weightB = Parameter(torch.Tensor(out_channels, rank, 1, 1))
        self.weightC = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weightA.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        n = self.rank
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weightB.data.uniform_(-stdv, stdv)

        n = self.in_channels * self.out_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weightC.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, rank={rank}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def forward(self, input):
        # print("B {}, A{}, C{}".format(self.weightB.size(), self.weightA.size(), self.weightC.size()))
        # print(self.extra_repr())
        # print(input[0][0][0])
        weightA = self.weightA.view(self.rank, -1)
        weightB = self.weightB.view(self.out_channels, -1)
        weight = weightB.matmul(weightA).view(self.weightC.size()) + self.weightC
        # exit(0)
        # print("B {}, A {}, C {}, W {}".format(weightB.data.cpu()[0][0][0], weightA.data.cpu()[0][0][0], self.weightC.data.cpu()[0][0][0], weight.data.cpu()[0][0][0]))
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # return F.conv2d(
        #     F.conv2d(input, self.weightA, None, self.stride, self.padding,                     self.dilation, self.groups),
        #     self.weightB, None, 1, 0, 1, 1) + \
        #     F.conv2d(input, self.weightC, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        # return F.conv2d(input, self.weightB.matmul(self.weightA) + self.weightC, self.bias, self.stride, self.padding, self.dilation, self.groups)

class CaffeLeNetSP(nn.Module):
    def __init__(self, ranks):
        super(CaffeLeNetSP, self).__init__()
        feature_layers = []
        # conv
        # feature_layers.append(conv_class(h, w, 1, 20, kernel_size=5))
        feature_layers.append(Conv2dsp(in_channels=1, out_channels=20, kernel_size=5, rank=ranks[0]))
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # conv
        # feature_layers.append(conv_class(h, w, 20, 50, kernel_size=5))
        feature_layers.append(Conv2dsp(in_channels=20, out_channels=50, kernel_size=5, rank=ranks[1]))
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*feature_layers)

        self.classifier = nn.Sequential(
            Linearsp(50 * 4 * 4, 500, rank=ranks[2]),
            nn.ReLU(inplace=True),
            Linearsp(500, 10, rank=ranks[3]),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 50 * 4 * 4)
        x = self.classifier(x)
        return x

    def set_weights(self, weights_list):
        i = 0
        for m in self.factorized_modules():
            weightA, weightB, weightC = weights_list[i]
            m.weightA.data.view(-1).copy_(weightA.view(-1))
            m.weightB.data.view(-1).copy_(weightB.view(-1))
            m.weightC.data.view(-1).copy_(weightC.view(-1))
            i += 1
    
    def factorized_modules(self):
        for module in self.modules():
            if isinstance(module, Conv2dsp) or isinstance(module, Linearsp):
                yield module

#######################################################################
#########################-Factorize Layer Design low rank-#############
#######################################################################

class Linearlr(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super(Linearlr, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        print("rank {}, in_features {}, out_features {}".format(rank, in_features, out_features))
        assert rank <= min(in_features, out_features)
        self.rank = rank
        self.weightA = Parameter(torch.Tensor(rank, in_features))
        self.weightB = Parameter(torch.Tensor(out_features, rank))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weightA.size(1))
        self.weightA.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.weightB.size(1))
        self.weightB.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # print("B {}, A{}, C{}".format(self.weightB.size(), self.weightA.size(), self.weightC.size()))
        weight = self.weightB.matmul(self.weightA)
        return F.linear(input, weight, self.bias)
        # return F.linear(input.matmul(self.weightA.t()), self.weightB, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Conv2dlr(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rank, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dlr, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        assert groups == 1, 'does not support grouped convolution yet'
        assert rank <= min(in_channels * kernel_size[0] * kernel_size[1], out_channels)
        self.rank = rank
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weightA = Parameter(torch.Tensor(rank, in_channels, *kernel_size))
        self.weightB = Parameter(torch.Tensor(out_channels, rank, 1, 1))
        self.weight_size = [out_channels, in_channels, *kernel_size]
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weightA.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        n = self.rank
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weightB.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, rank={rank}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def forward(self, input):
        # print("B {}, A{}, C{}".format(self.weightB.size(), self.weightA.size(), self.weightC.size()))
        # print(self.extra_repr())
        # print(input[0][0][0])
        weightA = self.weightA.view(self.rank, -1)
        weightB = self.weightB.view(self.out_channels, -1)
        weight = weightB.matmul(weightA).view(self.weight_size)
        # exit(0)
        # print("B {}, A {}, C {}, W {}".format(weightB.data.cpu()[0][0][0], weightA.data.cpu()[0][0][0], self.weightC.data.cpu()[0][0][0], weight.data.cpu()[0][0][0]))
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class CaffeLeNetLR(nn.Module):
    def __init__(self, ranks):
        super(CaffeLeNetLR, self).__init__()
        feature_layers = []
        # conv
        # feature_layers.append(conv_class(h, w, 1, 20, kernel_size=5))
        feature_layers.append(Conv2dlr(in_channels=1, out_channels=20, kernel_size=5, rank=ranks[0]))
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # conv
        # feature_layers.append(conv_class(h, w, 20, 50, kernel_size=5))
        feature_layers.append(Conv2dlr(in_channels=20, out_channels=50, kernel_size=5, rank=ranks[1]))
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*feature_layers)

        self.classifier = nn.Sequential(
            Linearlr(50 * 4 * 4, 500, rank=ranks[2]),
            nn.ReLU(inplace=True),
            Linearlr(500, 10, rank=ranks[3]),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 50 * 4 * 4)
        x = self.classifier(x)
        return x

    def set_weights(self, weights_list):
        i = 0
        for m in self.factorized_modules():
            weightA, weightB = weights_list[i]
            m.weightA.data.view(-1).copy_(weightA.view(-1))
            m.weightB.data.view(-1).copy_(weightB.view(-1))
            i += 1
    
    def factorized_modules(self):
        for module in self.modules():
            if isinstance(module, Conv2dlr) or isinstance(module, Linearlr):
                yield module


#######################################################################
#########################-Factorize Layer Design-######################
#######################################################################
            
class Linearsp_v2(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super(Linearsp_v2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        print("rank {}, in_features {}, out_features {}".format(rank, in_features, out_features))
        assert rank <= min(in_features, out_features)
        self.rank = rank
        self.weightA = Parameter(torch.zeros(rank, in_features))
        self.weightB = Parameter(torch.zeros(out_features, rank))
        self.weightC = Parameter(torch.zeros(out_features, in_features))
        if torch.cuda.is_available():
            self.eye = torch.eye(rank).cuda()
        else:
            self.eye = torch.eye(rank)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weightA.size(1))
        # self.weightA.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        # stdv = 1. / math.sqrt(self.weightB.size(1))
        # self.weightB.data.uniform_(-stdv, stdv)

        # stdv = 1. / math.sqrt(self.weightC.size(1))
        # self.weightC.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # print("B {}, A{}, C{}".format(self.weightB.size(), self.weightA.size(), self.weightC.size()))
        if self.rank == self.in_features:
            weight = self.weightB.matmul(self.weightA + self.eye) + self.weightC
        else:
            weight = (self.weightB + self.eye).matmul(self.weightA) + self.weightC
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Conv2dsp_v2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rank, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dsp_v2, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        assert groups == 1, 'does not support grouped convolution yet'
        assert rank <= min(in_channels * kernel_size[0] * kernel_size[1], out_channels)
        self.rank = rank
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weightA = Parameter(torch.zeros(rank, in_channels, *kernel_size))
        self.weightB = Parameter(torch.zeros(out_channels, rank, 1, 1))
        self.weightC = Parameter(torch.zeros(out_channels, in_channels, *kernel_size))
        if torch.cuda.is_available():
            self.eye = torch.eye(rank).cuda()
        else:
            self.eye = torch.eye(rank)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        # self.weightA.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        # n = self.rank
        # for k in self.kernel_size:
        #     n *= k
        # stdv = 1. / math.sqrt(n)
        # self.weightB.data.uniform_(-stdv, stdv)

        # n = self.in_channels * self.out_channels
        # for k in self.kernel_size:
        #     n *= k
        # stdv = 1. / math.sqrt(n)
        # self.weightC.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, rank={rank}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def forward(self, input):
        # print("B {}, A{}, C{}".format(self.weightB.size(), self.weightA.size(), self.weightC.size()))
        # print(self.extra_repr())
        # print(input[0][0][0])
        weightA = self.weightA.view(self.rank, -1)
        weightB = self.weightB.view(self.out_channels, -1)
        if self.rank == self.in_channels * self.kernel_size[0] * self.kernel_size[1]:
            weight = weightB.matmul(weightA + self.eye).view(self.weightC.size()) + self.weightC
        else:
            weight = (weightB + self.eye).matmul(weightA).view(self.weightC.size()) + self.weightC
        # exit(0)
        # print("B {}, A {}, C {}, W {}".format(weightB.data.cpu()[0][0][0], weightA.data.cpu()[0][0][0], self.weightC.data.cpu()[0][0][0], weight.data.cpu()[0][0][0]))
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # return F.conv2d(
        #     F.conv2d(input, self.weightA, None, self.stride, self.padding,                     self.dilation, self.groups),
        #     self.weightB, None, 1, 0, 1, 1) + \
        #     F.conv2d(input, self.weightC, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        # return F.conv2d(input, self.weightB.matmul(self.weightA) + self.weightC, self.bias, self.stride, self.padding, self.dilation, self.groups)

class CaffeLeNetSP_v2(nn.Module):
    def __init__(self, ranks):
        super(CaffeLeNetSP_v2, self).__init__()
        feature_layers = []
        # conv
        # feature_layers.append(conv_class(h, w, 1, 20, kernel_size=5))
        feature_layers.append(Conv2dsp_v2(in_channels=1, out_channels=20, kernel_size=5, rank=ranks[0]))
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # conv
        # feature_layers.append(conv_class(h, w, 20, 50, kernel_size=5))
        feature_layers.append(Conv2dsp_v2(in_channels=20, out_channels=50, kernel_size=5, rank=ranks[1]))
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*feature_layers)

        self.classifier = nn.Sequential(
            Linearsp_v2(50 * 4 * 4, 500, rank=ranks[2]),
            nn.ReLU(inplace=True),
            Linearsp_v2(500, 10, rank=ranks[3]),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 50 * 4 * 4)
        x = self.classifier(x)
        return x

    def set_weights(self, weights_list):
        i = 0

        for m in self.factorized_modules():
            weightA, weightB, weightC = weights_list[i]
            # print(weights_list[i])
            if not(weightA is None):
                # print("setA")
                m.weightA.data.view(-1).copy_(weightA.view(-1))

            if not(weightB is None):
                # print("setB")
                m.weightB.data.view(-1).copy_(weightB.view(-1))
            # m.weightC.data.view(-1).copy_(weightC.view(-1))
            i += 1
    
    def factorized_modules(self):
        for module in self.modules():
            if isinstance(module, Conv2dsp_v2) or isinstance(module, Linearsp_v2):
                yield module