import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import scipy.optimize as opt
import matplotlib.gridspec as gridspec


# Define the fit function.
def uncert(n, c_stat, c_syst):
  return np.sqrt(c_syst**2 + (c_stat**2) / n)

n_seg = np.linspace(5000,180000,100)

data = pd.read_csv("fullCovariancedataDTIOV123.csv")
data["NSegments"] = pd.to_numeric(data["NSegments"], errors='coerce')
data.dropna()
print(data.head(5))

mpl.rcParams['font.size'] = 5

# Define all relevant subsets of data.
data_DT_st12 = data.loc[(data['station'] == 1) | (data['station'] == 2)]
data_DT_st3 = data.loc[data['station'] == 3]
data_DT_st4 = data.loc[data['station'] == 4]
data_DT_wh0 = data.loc[data['wheel'] == 0]
data_DT_wh1 = data.loc[abs(data['wheel']) == 1]
data_DT_wh2 = data.loc[abs(data['wheel']) == 2]
data_DT_wh0_st123 = data.loc[(data['wheel'] == 0) & (data['station'] < 4)]
data_DT_wh1_st123 = data.loc[(abs(data['wheel']) == 1) & (data['station'] < 4)]
data_DT_wh2_st123 = data.loc[(abs(data['wheel']) == 2) & (data['station'] < 4)]
data_DT_wh0_st4 = data.loc[(data['wheel'] == 0) & (data['station'] == 4)]
data_DT_wh1_st4 = data.loc[(abs(data['wheel']) == 1) & (data['station'] == 4)]
data_DT_wh2_st4 = data.loc[(abs(data['wheel']) == 2) & (data['station'] == 4)]


fig = plt.figure()
fig.suptitle("Drift Tube APEs by DOF (2018 Data)")
fig.subplots_adjust(hspace=0.3, wspace=0.3)
gs = gridspec.GridSpec(2, 3)

# X
ax = fig.add_subplot(gs[0,0])

plt.scatter(data_DT_st12["NSegments"],
            10000*np.sqrt(data_DT_st12["xx"]),
            s=10, c='b', marker="o",label="Stations 1 and 2")
optimizedParameters_x1, pcov_x1 = opt.curve_fit(uncert, data_DT_st12["NSegments"],
                                                10000*np.sqrt(data_DT_st12["xx"]));
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_x1), label="fit", linewidth=2);

plt.scatter(data_DT_st3["NSegments"],
            10000*np.sqrt(data_DT_st3["xx"]),
            s=10, c='y', marker="o", label='Station 3')
plt.scatter(data_DT_st4["NSegments"],
            10000*np.sqrt(data_DT_st4["xx"]),
            s=10, c='r', marker="o", label='Station 4')
optimizedParameters_x3, pcov_x3 = opt.curve_fit(uncert, data_DT_st4["NSegments"],
                                                10000*np.sqrt(data_DT_st4["xx"]));
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_x3), label="fit3", linewidth=2);


# Quantiles
print(data_DT_st4.quantile(0.95))

plt.legend()
plt.title(r'$\sigma_{x}$')


# Y
ax = fig.add_subplot(gs[0,1])

plt.scatter(data_DT_wh0_st123["NSegments"],
            10000*np.sqrt(data_DT_wh0_st123["yy"]),
            s=10, c='b', marker="o", label="Wheel 0")
plt.scatter(data_DT_wh1_st123["NSegments"],
            10000*np.sqrt(data_DT_wh1_st123["yy"]),
            s=10, c='y', marker="o", label="Wheels +/- 1")
plt.scatter(data_DT_wh2_st123["NSegments"],
            10000*np.sqrt(data_DT_wh2_st123["yy"]),
            s=10, c='r', marker="o", label="Wheels +/- 2")

plt.legend()
plt.title(r'$\sigma_{y}$')


# Z
ax = fig.add_subplot(gs[0,2])

plt.scatter(data_DT_st12["NSegments"],
            10000*np.sqrt(data_DT_st12["zz"]),
            s=10, c='b', marker="o", label="Stations 1 and 2")
plt.scatter(data_DT_st3["NSegments"],
            10000*np.sqrt(data_DT_st3["zz"]),
            s=10, c='y', marker="o", label='Station 3')
plt.scatter(data_DT_st4["NSegments"],
            10000*np.sqrt(data_DT_st4["zz"]),
            s=10, c='r', marker="o", label='Station 4')

plt.legend()
plt.title(r'$\sigma_{z}$')



# PHI X
ax = fig.add_subplot(gs[1,0])

# AT: probably need to split by sector
plt.scatter(data_DT_st12["NSegments"],
            1000*np.sqrt(data_DT_st12["aa"]),
            s=10, c='b', marker="o", label="Stations 1 and 2")
plt.scatter(data_DT_st3["NSegments"],
            1000*np.sqrt(data_DT_st3["aa"]),
            s=10, c='y', marker="o", label='Station 3')
plt.scatter(data_DT_st4["NSegments"],
            1000*np.sqrt(data_DT_st4["aa"]),
            s=10, c='r', marker="o", label='Station 4')

plt.legend()
plt.title(r'$\sigma_{\phi_x}$')



# PHI Y
ax = fig.add_subplot(gs[1,1])

plt.scatter(data_DT_st12["NSegments"],
            1000*np.sqrt(data_DT_st12["bb"]),
            s=10, c='b', marker="o", label="Stations 1 and 2")
plt.scatter(data_DT_st3["NSegments"],
            1000*np.sqrt(data_DT_st3["bb"]),
            s=10, c='y', marker="o", label='Station 3')
plt.scatter(data_DT_st4["NSegments"],
            1000*np.sqrt(data_DT_st4["bb"]),
            s=10, c='r', marker="o", label='Station 4')

plt.semilogy()
plt.legend()
plt.title(r'$\sigma_{\phi_y}$')


# PHI Z
ax = fig.add_subplot(gs[1,2])

plt.scatter(data_DT_st12["NSegments"],
            1000*np.sqrt(data_DT_st12["cc"]),
            s=10, c='b', marker="o", label="Stations 1 and 2")
plt.scatter(data_DT_st3["NSegments"],
            1000*np.sqrt(data_DT_st3["cc"]),
            s=10, c='y', marker="o", label='Station 3')
plt.scatter(data_DT_st4["NSegments"],
            1000*np.sqrt(data_DT_st4["cc"]),
            s=10, c='r', marker="o", label='Station 4')

plt.legend()
plt.title(r'$\sigma_{\phi_z}$')

plt.savefig("uncertainty_curves_allDOF.png")
plt.savefig("uncertainty_curves_allDOF.pdf")
