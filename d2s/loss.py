import torch
import torch.nn.functional as F


def mse_loss(pred, target):
	return F.mse_loss(pred, target)


def w1D_loss(x, y):
	x = x.view(-1, 1)
	y = y.view(-1, 1)

	a = torch.cdist(x, y, p=1).mean()  # E|X-Y|

	# E|X-X'| where X' ≠ X
	n = x.size(0)
	mask = 1 - torch.eye(n, device=x.device)
	b = (torch.cdist(x, x, p=1) * mask).sum() / (n * (n - 1))

	m = y.size(0)
	mask = 1 - torch.eye(m, device=y.device)
	c = (torch.cdist(y, y, p=1) * mask).sum() / (m * (m - 1))

	return 2 * a - b - c

def kl_loss(v_e, t_e):
	v_e = F.log_softmax(v_e, dim=0)
	t_e = F.softmax(t_e, dim=0)
	kl_loss = F.kl_div(v_e, t_e, reduction='sum')

	return kl_loss


def huber_loss(pred, target, delta=0.05):
	err = pred - target
	abs_err = err.abs()
	quad = torch.clamp(abs_err, max=delta)
	lin = abs_err - quad
	return torch.mean(0.5 * quad ** 2 + delta * lin)
