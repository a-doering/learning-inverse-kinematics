import csv

from matplotlib import pyplot as plt


LOG_FILE = "losses.csv"


# Read Log File
################

train_losses = []
val_losses = []
with open(LOG_FILE, newline='') as file:
    for row in csv.reader(file):
        train_losses.append(float(row[0]))
        val_losses.append(float(row[1]))


# Plot Losses
##############

fig, ax = plt.subplots()
ax.plot(train_losses, label="Train Loss")
ax.plot(val_losses, label="Val Loss")
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True)
plt.show()
