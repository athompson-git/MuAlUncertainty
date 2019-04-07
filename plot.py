import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt

# Define the fit function.
def uncert(n, c_stat, c_syst):
  return np.sqrt(c_syst**2 + (c_stat**2) / n)

n_seg = np.linspace(5000,180000,100)

data = pd.read_csv("fullCovariancedataDTIOV123.csv")
data["NSegments"] = pd.to_numeric(data["NSegments"], errors='coerce')
data.dropna()
print(data.head(5))


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

# X
plt.scatter(data_DT_st12["NSegments"],
            10000*np.sqrt(data_DT_st12["xx"]),
            c='b', marker="o", label="Stations 1 and 2")
optimizedParameters_x1, pcov_x1 = opt.curve_fit(uncert, data_DT_st12["NSegments"],
                                                10000*np.sqrt(data_DT_st12["xx"]));
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_x1), label="fit");

plt.scatter(data_DT_st3["NSegments"],
            10000*np.sqrt(data_DT_st3["xx"]),
            c='y', marker="o", label='Station 3')
plt.scatter(data_DT_st4["NSegments"],
            10000*np.sqrt(data_DT_st4["xx"]),
            c='r', marker="o", label='Station 4')
optimizedParameters_x3, pcov_x3 = opt.curve_fit(uncert, data_DT_st4["NSegments"],
                                                10000*np.sqrt(data_DT_st4["xx"]));
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_x3), label="fit3");

plt.legend()
plt.title("DT dx")
plt.savefig("xx_DT.png")
plt.clf()


# Y
plt.scatter(data_DT_wh0_st123["NSegments"],
            10000*np.sqrt(data_DT_wh0_st123["yy"]),
            c='b', marker="o", label="Wheel 0")
plt.scatter(data_DT_wh1_st123["NSegments"],
            10000*np.sqrt(data_DT_wh1_st123["yy"]),
            c='y', marker="o", label="Wheels +/- 1")
plt.scatter(data_DT_wh2_st123["NSegments"],
            10000*np.sqrt(data_DT_wh2_st123["yy"]),
            c='r', marker="o", label="Wheels +/- 2")

plt.legend()
plt.title("DT dy")
plt.savefig("yy_DT.png")
plt.clf()


# Z
plt.scatter(data_DT_st12["NSegments"],
            10000*np.sqrt(data_DT_st12["zz"]),
            c='b', marker="o", label="Stations 1 and 2")
plt.scatter(data_DT_st3["NSegments"],
            10000*np.sqrt(data_DT_st3["zz"]),
            c='y', marker="o", label='Station 3')
plt.scatter(data_DT_st4["NSegments"],
            10000*np.sqrt(data_DT_st4["zz"]),
            c='r', marker="o", label='Station 4')

plt.legend()
plt.title("DT dz")
plt.savefig("zz_DT.png")

plt.clf()


# PHI X
# AT: probably need to split by sector
plt.scatter(data_DT_st12["NSegments"],
            1000*np.sqrt(data_DT_st12["aa"]),
            c='b', marker="o", label="Stations 1 and 2")
plt.scatter(data_DT_st3["NSegments"],
            1000*np.sqrt(data_DT_st3["aa"]),
            c='y', marker="o", label='Station 3')
plt.scatter(data_DT_st4["NSegments"],
            1000*np.sqrt(data_DT_st4["aa"]),
            c='r', marker="o", label='Station 4')

plt.legend()
plt.title("DT phi_x")
plt.savefig("aa_DT.png")

plt.clf()


# PHI Y
plt.scatter(data_DT_st12["NSegments"],
            1000*np.sqrt(data_DT_st12["bb"]),
            c='b', marker="o", label="Stations 1 and 2")
plt.scatter(data_DT_st3["NSegments"],
            1000*np.sqrt(data_DT_st3["bb"]),
            c='y', marker="o", label='Station 3')
plt.scatter(data_DT_st4["NSegments"],
            1000*np.sqrt(data_DT_st4["bb"]),
            c='r', marker="o", label='Station 4')

plt.semilogy()
plt.legend()
plt.title("DT phi_y")
plt.savefig("bb_DT.png")
plt.clf()


# PHI Z
plt.scatter(data_DT_st12["NSegments"],
            1000*np.sqrt(data_DT_st12["cc"]),
            c='b', marker="o", label="Stations 1 and 2")
plt.scatter(data_DT_st3["NSegments"],
            1000*np.sqrt(data_DT_st3["cc"]),
            c='y', marker="o", label='Station 3')
plt.scatter(data_DT_st4["NSegments"],
            1000*np.sqrt(data_DT_st4["cc"]),
            c='r', marker="o", label='Station 4')

plt.legend()
plt.title("DT phi_z")
plt.savefig("cc_DT.png")


plt.legend()
plt.show()

