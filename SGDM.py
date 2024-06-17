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

    '''实现SGDM算法'''
    # 实例化模型和SGDM优化器
    net_sgdm = models.resnet18(pretrained=False, num_classes=10).to(device)
    optimizer_sgdm = optim.SGD(net_sgdm.parameters(), lr=0.001, momentum=0.9)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 记录训练时间
    start_time = time.time()
    #进行训练和测试
    for epoch in range(epoch_num):
        loss = train(net_sgdm, trainloader, optimizer_sgdm, criterion, device=device)
        accuracy = test(net_sgdm, testloader, device=device)
        losses.append(loss)
        accuracies.append(accuracy)
        print(f"Epoch {epoch + 1}: Loss: {loss}, Accuracy:{accuracy}")
    adam_time = time.time() - start_time
    print(f"sgdm_time:{adam_time}")

    torch.save(net_sgdm, 'adai_adlr_model')

    # 保存准确率到文件
    with open('SGDM_accuracy_100_epochs_lr=0.001.pkl', 'wb') as file:
        pickle.dump(accuracies, file)

    # 保存损失到文件
    with open('SGDM_loss_100_epochs_lr=0.001.pkl', 'wb') as file:
        pickle.dump(losses, file)

