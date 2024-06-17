import matplotlib.pyplot as plt
import pickle
adam_file = open('adam_100_epochs_lr=0.001.pkl', 'rb')
adai_file = open('adai_100_epochs_lr=1.0.pkl', 'rb')
adai_adaptivelr_file = open('adai_adlr_100_epochs_lr=0.001.pkl', 'rb')
sgdw_file = open('sgdm_100_epochs_lr=0.001.pkl', 'rb')
y_adai_accuracy = pickle.load(adai_file)
y_adam_accuracy = pickle.load(adam_file)
y_adai_adaptivelr_accuracy = pickle.load(adai_adaptivelr_file)
y_sgdw_accuracy = pickle.load(sgdw_file)

epoch = range(len(y_adam_accuracy))
plt.figure()
# 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('epoch')    # x轴标签
plt.ylabel('accuracy(%)')     # y轴标签

plt.plot(epoch, y_adam_accuracy, linewidth=1, linestyle="solid", label="adam", color='red')
plt.plot(epoch, y_adai_accuracy, linewidth=1, linestyle="solid", label="adai", color='blue')
plt.plot(epoch, y_adai_adaptivelr_accuracy, linewidth=1, linestyle="solid", label="adai_adlr", color='purple')
plt.plot(epoch, y_sgdw_accuracy, linewidth=1, linestyle="solid", label="sgdw", color='green')
plt.legend()
plt.title('accuracy curve')
plt.show()
