import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/content/drive/MyDrive/ai_haiku/ai_haiku_last/generator_loss_log.csv")

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(df["epoch"], df["train_loss"],
        marker='o', label='train')

ax.plot(df["epoch"], df["val_loss"],
        marker='s', label='validation')

best_epoch = 13
best_val = df.loc[df.epoch==best_epoch, "val_loss"].values[0]

ax.plot(best_epoch, best_val,
        marker='^', markersize=10,
        label='best')

ax.set_xlabel('epoch', fontsize=16)
ax.set_ylabel('loss', fontsize=16)
ax.legend()

plt.show()