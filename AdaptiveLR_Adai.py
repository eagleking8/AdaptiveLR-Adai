import torch
import torch.optim as optim

'''实现结合自适应学习率的Adai算法'''
class AdaiAdaptiveLR(optim.Optimizer):
    def __init__(self, params, lr=0.001, beta0=0.1, beta2=0.99, epsilon=1e-3):
        defaults = dict(lr=lr, beta0=beta0, beta2=beta2, epsilon=epsilon)
        super(AdaiAdaptiveLR, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # 记录所有参数的个数
        param_size = 0

        # 累加所有二阶梯度
        v_hat_sum = 0.

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_size += p.numel()
                grad = p.grad.data
                state = self.state[p]

                # 初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['beta1_prod'] = torch.ones_like(p.data, memory_format=torch.preserve_format)

                v = state['v']
                beta0, beta2, epsilon = group['beta0'], group['beta2'], group['epsilon']
                state['step'] += 1

                # 更新v_t
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 调整为v_hat
                v_hat = v / (1 - beta2 ** state['step'])

                v_hat_sum += v_hat.sum()

        # 计算v_mean的均值
        v_mean = v_hat_sum / param_size

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                m = state['m']

                epsilon, beta0, beta2 = group['epsilon'], group['beta0'], group['beta2']

                v_hat = state['v'] / (1 - beta2 ** state['step'])

                # 自适应调整动量因子beta1_t
                beta1_t = (1 - beta0 / v_mean * v_hat).clamp(0, 1 - epsilon)
                state['beta1_prod'].mul_(beta1_t)

                # 更新m_t
                m.mul_(beta1_t).add_(grad * (1 - beta1_t))

                # 偏差校正
                m_hat = m / (1 - state['beta1_prod'])

                # 计算自适应学习率
                adaptive_lr = group['lr'] / (torch.sqrt(v_hat) + 1e-8)

                # 更新参数
                p.data.add_(- (m_hat * adaptive_lr))

        return loss



