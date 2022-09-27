import torch
from torch import nn
from torch.nn import functional as F
import pair_wise_distance_cuda as pdCuda


class PairwiseDistFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        # pFea, spFea, initSpIdx, nSpW, nSpH
        pFea = args[0]
        spFea = args[1]
        initSpIdx = args[2]

        ctx.nSpW = args[3]
        ctx.nSpH = args[4]

        with torch.cuda.device_of(pFea):
            output = pFea.new(pFea.shape[0], 9, pFea.shape[-1]).zero_()
            ctx.save_for_backward(pFea, spFea, initSpIdx)

            pdCuda.forward(pFea.contiguous(),
                           spFea.contiguous(),
                           initSpIdx.contiguous(),
                           output.contiguous(),
                           ctx.nSpW, ctx.nSpH)
        return output

    @staticmethod
    def backward(ctx, *args):
        dist_matrix_grad = args[0]
        pFea, spFea, spIdx = ctx.saved_tensors

        with torch.cuda.device_of(pFea):
            pFeaGrad = torch.zeros_like(pFea)
            spFeaGrad = torch.zeros_like(spFea)

            pdCuda.backward(dist_matrix_grad.contiguous(),
                            pFea.contiguous(),
                            spFea.contiguous(),
                            spIdx.contiguous(),
                            pFeaGrad,
                            spFeaGrad,
                            ctx.nSpW, ctx.nSpH)

        return pFeaGrad, spFeaGrad, None, None, None


class PairwiseDist(nn.Module):
    def __init__(self, requires_grad=True):
        super(PairwiseDist, self).__init__()
        self.requires_grad = requires_grad

    def forward(self, pFea, spFea, initSpIdx, nSpW, nSpH):
        return PairwiseDistFunction.apply(pFea, spFea, initSpIdx, nSpW, nSpH)


def naive_pair_wise_dist(pix, spix, idx, n_spix_w, n_spix_h):
    device = pix.device
    myOutput = torch.ones([2, 9, 81]).double().to(device) * 1e16
    ba, ch, pi = pix.shape

    for b in range(ba):
        for p in range(pi):
            pix_v = pix[b, :, p]
            sp_i = idx[b, p]
            sp_i_x = sp_i % n_spix_w
            sp_i_y = torch.div(sp_i, n_spix_w, rounding_mode='floor')
            for i in range(9):
                refuse = [(sp_i_x == 0 and (i % 3) == 0)]
                refuse.append((sp_i_x == (n_spix_w - 1) and (i % 3) == 2))
                refuse.append((sp_i_y == 0 and (i // 3) == 0))
                refuse.append((sp_i_y == (n_spix_h - 1) and (i // 3) == 2))

                if not any(refuse):
                    offset_x = i % 3 - 1
                    offset_y = (i // 3 - 1) * n_spix_w
                    s = int(sp_i + offset_y + offset_x)
                    myOutput[b, i, p] = (pix_v - spix[b, :, s]).pow(2).sum()
    return myOutput


def check():
    pFea = F.normalize(torch.randn(2, 16, 81).double().cuda(), dim=1)
    spFea = F.normalize(torch.randn(2, 16, 9).double().cuda(), dim=1)
    initSpIdx = torch.randint(0, 9, (2, 81)).double().cuda()
    nSpW = 3
    nSpH = 3

    pFea.requires_grad = True
    spFea.requires_grad = True
    Pair = PairwiseDist()
    Func = PairwiseDistFunction.apply

    res = torch.autograd.gradcheck(Func, (pFea, spFea, initSpIdx, nSpW, nSpH), eps=1e-4, raise_exception=False)

    o = Pair(pFea, spFea, initSpIdx, nSpW, nSpH)
    # o = torch.exp(-o)
    o.sum().backward()

    pGradCuda = pFea.grad
    spGradCuda = spFea.grad

    pFea.grad.zero_()
    spFea.grad.zero_()

    naive_o = naive_pair_wise_dist(pFea, spFea, initSpIdx, nSpW, nSpH)
    # naive_o = torch.exp(-naive_o)
    naive_o.sum().backward()

    pGradTorch = pFea.grad
    spGradTorch = spFea.grad
    print('check grad is ', res)
    print("output diff", torch.abs(o - naive_o).mean().cpu().item())
    print("pix grad diff", torch.abs(pGradCuda - pGradTorch).mean().cpu().item())
    print("spix grad diff", torch.abs(spGradCuda - spGradTorch).mean().cpu().item())


if __name__ == '__main__':
    check()
