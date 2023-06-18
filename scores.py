
from tabulate import tabulate
import numpy as np



scores_acc = np.load("scores_acc.npy")
scores_bal_acc = np.load("scores_bal_acc.npy")
scores_rec = np.load("scores_rec.npy")
scores_prec = np.load("scores_prec.npy")


print("scores_acc")

table = tabulate(np.mean(scores_acc, axis=-1),
                  tablefmt='grid',
                  headers=["NWC", "CWC","DWC", "RC", "SC"],
                  showindex=["dataset1", "dataset2", "dataset3", "dataset4", "dataset5", "dataset6"])

print(table)


print("scores_bal_acc")

table = tabulate(np.mean(scores_bal_acc, axis=-1),
                  tablefmt='grid',
                  headers=["NWC", "CWC","DWC", "RC", "SC"],
                  showindex=["dataset1", "dataset2", "dataset3", "dataset4", "dataset5", "dataset6"])

print(table)

print("scores_prec")

table = tabulate(np.mean(scores_prec, axis=-1),
                  tablefmt='grid',
                  headers=["NWC", "CWC","DWC", "RC", "SC"],
                  showindex=["dataset1", "dataset2", "dataset3", "dataset4", "dataset5", "dataset6"])

print(table)



table = tabulate(np.mean(scores_rec, axis=-1),
                  tablefmt='grid',
                  headers=["NWC", "CWC","DWC", "RC", "SC"],
                  showindex=["dataset1", "dataset2", "dataset3", "dataset4", "dataset5", "dataset6"])

print(table)