# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 13:53:04 2025

@author: socce
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Plot
fig1 = plt.figure(figsize=(12, 8))
ax1 = fig1.add_subplot(111)
ax1.plot(loss_mat[100:], linestyle='-', linewidth=3, label="Without first 100 epochs")

ax1.set_xlabel("Training epochs", fontsize=16)
ax1.set_ylabel("Training error", fontsize=16)

ax1.tick_params(axis='x', labelsize=15)  # X-axis tick label font size
ax1.tick_params(axis='y', labelsize=15)  # Y-axis tick label font size

ax1.legend(fontsize=16)  # Proper legend handling
plt.tight_layout()
plt.show()