import csv

from matplotlib import pyplot as plt


LOG_FILE = "log/losses.csv"

# (column in the .csv file, label in the plot)
TRAIN_LOSS = (0, "Train Loss")
TRAIN_FORWARD_FIT = (1, "Train Forward L2 Fit")
TRAIN_FORWARD_MMD = (2, "Train Forward MMD")
TRAIN_BACKWARD_MMD = (3, "Train Backward MMD")
TRAIN_RECONSTRUCTION = (4, "Train Reconstruction L2 Fit")
VAL_LOSS = (5, "Val Loss")
VAL_FORWARD_FIT = (6, "Val Forward L2 Fit")
VAL_FORWARD_MMD = (7, "Val Forward MMD")
VAL_BACKWARD_MMD = (8, "Val Backward MMD")
VAL_RECONSTRUCTION = (9, "Val Reconstruction L2 Fit")

# define what data should be plotted
PLOT = [TRAIN_LOSS, VAL_LOSS]


# Read Log File
################

data = {index: [] for index, _ in PLOT}

with open(LOG_FILE, newline='') as file:
    for row in csv.reader(file):
        for index, _ in PLOT:
            data[index].append(float(row[index]))


# Plot Losses
##############

fig, ax = plt.subplots()

for index, label in PLOT:
    ax.plot(data[index], label=label)

ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.set_xlabel("Epoch")
ax.legend()
ax.grid(True)
plt.show()
