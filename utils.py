import torch
import torchvision
import torchvision.transforms as transforms

'''数据集准备'''
def load_data():
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # CIFAR-10 训练集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    # CIFAR-10 测试集
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    return trainloader, testloader

'''定义和训练测试函数'''
# 训练函数
def train(net, trainloader, optimizer, criterion, device = "cuda"):
    net.train()
    running_loss = 0.0
    for data, label in trainloader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()  # 清空梯度
        output = net(data)  # 前向传播
        loss = criterion(output, label)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        running_loss += loss.item()  # 累加损失
    return running_loss / len(trainloader) # 返回这个epoch的平均损失

# 测试函数
def test(net, testloader, device = "cuda"):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # 关闭梯度计算
        for data, label in testloader:
            data, label = data.to(device), label.to(device)
            output = net(data)  # 前向传播
            _, predicted = torch.max(output.data, 1)  # 取预测值
            total += label.size(0)
            correct += (predicted == label).sum().item()  # 累加正确预测的数量
    return 100 * correct / total # 返回准确率