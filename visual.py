import matplotlib.pyplot as plt
import pickle
adam_file = open('Adam_accuracy_200_epochs_lr=0.001_with_scheduler_data_augmentation.pkl', 'rb')
adai_file = open('Adai_unofficail_accuracy_200_epochs_lr=1.0_with_data_augmentation_scheduler.pkl', 'rb')
# adai05_file = open("Adai_accuracy_200_epochs_lr=0.5.pkl", 'rb')
adai_adaptivelr_file = open('Adlr_Adai_unofficail_accuracy_200_epochs_lr=0.001_with_data_augmentation_scheduler.pkl', 'rb')
sgdm_file = open('SGDM_accuracy_200_epochs_lr=0.1_with_scheduler_data_augmentation.pkl', 'rb')
adai_official_file = open('Adai_officail_accuracy_200_epochs_lr=1.0_with_data_augmentation_scheduler.pkl', 'rb')
# adai05_official_file = open('Adai_official_accuracy_200_epochs_lr=0.5.pkl', 'rb')
#
y_adai_accuracy = pickle.load(adai_file)[-100:]
# y_adai05_accuracy = pickle.load(adai05_file)
y_adam_accuracy = pickle.load(adam_file)[-100:]
y_adai_adaptivelr_accuracy = pickle.load(adai_adaptivelr_file)[-100:]
y_sgdm_accuracy = pickle.load(sgdm_file)[-100:]
y_adai_official_accuracy = pickle.load(adai_official_file)[-100:]
# y_adai05_official_accuracy = pickle.load((adai05_official_file))

epoch = range(100, 200)

plt.figure()
# 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('epoch')    # x轴标签
plt.ylabel('accuracy(%)')     # y轴标签

plt.plot(epoch, y_adam_accuracy, linewidth=1, linestyle="solid", label="adam", color='red')
plt.plot(epoch, y_adai_accuracy, linewidth=1, linestyle="solid", label="adai lr=1.0", color='blue')
plt.plot(epoch, y_adai_adaptivelr_accuracy, linewidth=1, linestyle="solid", label="adai_adlr", color='purple')
plt.plot(epoch, y_sgdm_accuracy, linewidth=1, linestyle="solid", label="sgdw", color='green')
# plt.plot(epoch, y_adai05_accuracy, linewidth=1, linestyle="solid", label="adai lr=0.5", color='pink')
plt.plot(epoch, y_adai_official_accuracy, linewidth=1, linestyle="solid", label="adai official lr=1.0", color='black')
# plt.plot(epoch, y_adai05_official_accuracy, linewidth=1, linestyle="solid", label="adai official lr=0.5", color='orange')

plt.legend()
plt.title('accuracy curve')
plt.show()

adam_loss_file = open('Adam_loss_200_epochs_lr=0.001_with_scheduler_data_augmentation.pkl', 'rb')
adai_loss_file = open('Adai_unofficial_loss_200_epochs_lr=1.0_with_data_augmentation_scheduler.pkl', 'rb')
# adai05_loss_file = open("Adai_loss_200_epochs_lr=0.5.pkl", 'rb')
adai_adaptivelr_loss_file = open('Adlr_Adai_unofficial_loss_200_epochs_lr=0.001_with_data_augmentation_scheduler.pkl', 'rb')
sgdm_loss_file = open('SGDM_loss_200_epochs_lr=0.1_with_scheduler_data_augmentation.pkl', 'rb')
adai_official_loss_file = open('Adai_official_loss_200_epochs_lr=1.0_with_data_augmentation_scheduler.pkl', 'rb')
# adai05_official_loss_file = open('Adai_official_loss_200_epochs_lr=0.5.pkl', 'rb')

y_adai_loss = pickle.load(adai_loss_file)[-100:]
# y_adai05_loss = pickle.load(adai05_loss_file)
y_adam_loss = pickle.load(adam_loss_file)[-100:]
y_adai_adaptivelr_loss = pickle.load(adai_adaptivelr_loss_file)[-100:]
y_sgdm_loss = pickle.load(sgdm_loss_file)[-100:]
y_adai_official_loss = pickle.load(adai_official_loss_file)[-100:]
# y_adai05_official_loss = pickle.load(adai05_official_loss_file)

plt.figure()
# 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('epoch')    # x轴标签
plt.ylabel('loss')     # y轴标签

plt.plot(epoch, y_adam_loss, linewidth=1, linestyle="solid", label="adam", color='red')
plt.plot(epoch, y_adai_loss, linewidth=1, linestyle="solid", label="adai lr=1.0", color='blue')
plt.plot(epoch, y_adai_adaptivelr_loss, linewidth=1, linestyle="solid", label="adai_adlr", color='purple')
plt.plot(epoch, y_sgdm_loss, linewidth=1, linestyle="solid", label="sgdw", color='green')
# plt.plot(epoch, y_adai05_loss, linewidth=1, linestyle="solid", label="adai lr=0.5", color='pink')
plt.plot(epoch, y_adai_official_loss, linewidth=1, linestyle="solid", label="adai official lr=1.0", color='black')
# plt.plot(epoch, y_adai05_official_loss, linewidth=1, linestyle="solid", label="adai official lr=0.5", color='orange')
plt.legend()
plt.title('loss curve')
plt.show()