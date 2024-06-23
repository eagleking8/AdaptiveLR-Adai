import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
import torchvision.models as models

import adai_optim
from utils import load_data,train,test
import Adai
import AdaptiveLR_Adai

if __name__ == '__main__':
    epoch_num = 200
    accuracies = []
    losses = []

    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 加载数据
    trainloader, testloader = load_data()

    # 实例化ResNet-18模型
    net = models.resnet18(pretrained=False, num_classes=10).to(device)
    if device == 'cuda' and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    optimizer_name = "SGDM"
    optimizer = ""

    # 选择并实例化优化器
    if optimizer_name == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0)
    elif optimizer_name == "SGDM":
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    elif optimizer_name == "Adai_officail":
        optimizer = adai_optim.Adai(net.parameters(), lr=1.0, betas=(0.1, 0.99), eps=1e-03, weight_decay=0, decoupled=False)
    elif optimizer_name == "Adai_unofficail":
        optimizer = Adai.Adai_unofficial(net.parameters(), lr=1.0, beta0=0.1, beta2=0.99, epsilon=1e-03)
    elif optimizer_name == "AdaptiveLR_Adai":
        optimizer = AdaptiveLR_Adai.AdaiAdaptiveLR(net.parameters(), lr=0.001, beta0=0.1, beta2=0.99, epsilon=1e-3)

    # 实例化损失函数
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # 加入学习率调度
    def lambda_lr(epoch): return 0.1 ** (epoch // 80)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    # 记录训练时间
    # start_time = time.time()
    #进行训练和测试
    for epoch in range(epoch_num):
        loss = train(net, trainloader, optimizer, criterion, device=device)
        accuracy = test(net, testloader, device=device)
        losses.append(loss)
        accuracies.append(accuracy)
        scheduler.step()
        print(f"Epoch {epoch + 1}: Loss: {loss}, Accuracy:{accuracy}")
    # adam_time = time.time() - start_time
    # print(f"adai_time:{adam_time}")

    # 保存准确率到文件
    with open('SGDM_accuracy_200_epochs_lr=0.1_with_scheduler_data_augmentation.pkl', 'wb') as file:
        pickle.dump(accuracies, file)

    # 保存损失到文件
    with open('SGDM_loss_200_epochs_lr=0.1_with_scheduler_data_augmentation.pkl', 'wb') as file:
        pickle.dump(losses, file)

    # torch.save(net, 'adai_model')