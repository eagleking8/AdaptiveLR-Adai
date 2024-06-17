import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import time
import torchvision.models as models
from utils import load_data,train,test

if __name__ == '__main__':
    epoch_num = 5
    accuracies = []
    losses = []

    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 加载数据
    trainloader, testloader = load_data()

    '''实现Adai算法'''
    class Adai(optim.Optimizer):
        def __init__(self, params, lr=1.0, beta0=0.1, beta2=0.99, epsilon=1e-3):
            defaults = dict(lr=lr, beta0=beta0, beta2=beta2, epsilon=epsilon)
            super(Adai, self).__init__(params, defaults)

        def step(self, closure=None):
            loss = None
            if closure is not None:
                loss = closure()

            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    state = self.state[p]

                    # 初始化状态
                    if len(state) == 0:
                        state['step'] = 0
                        state['m'] = torch.zeros_like(p.data)
                        state['v'] = torch.zeros_like(p.data)
                        state['beta1_prod'] = torch.ones_like(p.data)

                    m, v = state['m'], state['v']
                    beta0, beta2, epsilon = group['beta0'], group['beta2'], group['epsilon']

                    state['step'] += 1

                    # 更新v_t
                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    state['v'] = v
                    v_hat = v / (1 - beta2 ** state['step'])

                    # 计算v_hat的均值
                    v_mean = v_hat.mean()

                    # 自适应调整动量因子beta1_t
                    beta1_t = (1 - beta0 / v_mean * v_hat).clamp(0, 1 - epsilon)
                    state['beta1_prod'].mul_(beta1_t)

                    # 更新m_t
                    m.mul_(beta1_t).add_(grad * (1 - beta1_t))
                    state['m'] = m

                    # 偏差校正
                    m_hat = m / (1 - state['beta1_prod'])

                    # 更新参数
                    p.data.add_(m_hat, alpha=-group['lr'])

            return loss


    # 实例化模型和Adai优化器
    net_adai = models.resnet18(pretrained=False, num_classes=10).to(device)
    optimizer_adai = Adai(net_adai.parameters(), lr=1.0)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 记录训练时间
    start_time = time.time()
    #进行训练和测试
    for epoch in range(epoch_num):
        loss = train(net_adai, trainloader, optimizer_adai, criterion, device=device)
        accuracy = test(net_adai, testloader, device=device)
        losses.append(loss)
        accuracies.append(accuracy)
        print(f"Epoch {epoch + 1}: Loss: {loss}, Accuracy:{accuracy}")
    adam_time = time.time() - start_time
    print(f"adam_time:{adam_time}")

    torch.save(net_adai, 'adai_model')

    # 保存准确率到文件
    with open('Adai_accuracy_100_epochs_lr=1.0.pkl', 'wb') as file:
        pickle.dump(accuracies, file)

    # 保存损失到文件
    with open('Adai_loss_100_epochs_lr=1.0.pkl', 'wb') as file:
        pickle.dump(losses, file)

