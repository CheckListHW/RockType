import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure()

ax = fig.add_subplot(111)

gs = gridspec.GridSpec(1,3)
ax.set_position(gs[1].get_position(fig))
ax.set_subplotspec(gs[1])              # only necessary if using tight_layout()

fig.add_subplot(gs[2])

fig.tight_layout()                       # not strictly part of the question

plt.show()