import pickle
import torch
import torch.nn as nn
import time
import torchvision.models as models
from utils import load_data,train,test
import adai_optim

if __name__ == '__main__':
    epoch_num = 200
    accuracies = []
    losses = []

    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 加载数据
    trainloader, testloader = load_data()

    '''实现Adai算法'''
    # 实例化模型和Adam优化器
    net = models.resnet18(pretrained=False, num_classes=10).to(device)
    Adai = adai_optim.Adai(net.parameters(), lr=0.5, betas=(0.1, 0.99), eps=1e-03, weight_decay=0, decoupled=False)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 记录训练时间
    start_time = time.time()
    #进行训练和测试
    for epoch in range(epoch_num):
        loss = train(net, trainloader, Adai, criterion, device=device)
        accuracy = test(net, testloader, device=device)
        losses.append(loss)
        accuracies.append(accuracy)
        print(f"Epoch {epoch + 1}: Loss: {loss}, Accuracy:{accuracy}")
    adam_time = time.time() - start_time
    print(f"adai_official_time:{adam_time}")

    # 保存准确率到文件
    with open('Adai_official_accuracy_200_epochs_lr=0.5.pkl', 'wb') as file:
        pickle.dump(accuracies, file)

    # 保存损失到文件
    with open('Adai_official_loss_200_epochs_lr=0.5.pkl', 'wb') as file:
        pickle.dump(losses, file)

    torch.save(net, 'adai_official_0.5_model')