import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt
import matplotlib.gridspec as gridspec


# Define the fit function.
def uncert(n, c_stat, c_syst):
    return np.sqrt(c_syst ** 2 + (c_stat ** 2) / n)


# Create an array of segments for the fit functions to use as arguments
# (for plotting purposes only).
n_seg = np.linspace(5, 180, 100)

# Matplotlib settings
mpl.rcParams['font.size'] = 4
mpl.rcParams['lines.markersize'] = 2

data = pd.read_csv("fullCovariancedataDTIOV123.csv")

# Clean the data.
data["NSegments"] = pd.to_numeric(data["NSegments"], errors='coerce')
data = data[np.isfinite(data['NSegments'])]
data['NSegments'] = 0.001 * data['NSegments']  # convert to 1000's of segments

data['xx'] = 10000 * np.sqrt(data['xx'])  # convert to microns
data['yy'] = 10000 * np.sqrt(data['yy'])
data['zz'] = 10000 * np.sqrt(data['zz'])
data['aa'] = 1000 * np.sqrt(data['aa'])  # convert to mrad
data['bb'] = 1000 * np.sqrt(data['bb'])
data['cc'] = 1000 * np.sqrt(data['cc'])
print(data.head(5))

# Define all relevant geometric subgroups.
data_DT_st1 = data.loc[data['station'] == 1]
data_DT_st2 = data.loc[data['station'] == 2]
data_DT_st12 = data.loc[(data['station'] == 1) | (data['station'] == 2)]
data_DT_st3 = data.loc[data['station'] == 3]
data_DT_st4 = data.loc[data['station'] == 4]
data_DT_wh0 = data.loc[data['wheel'] == 0]
data_DT_wh1 = data.loc[abs(data['wheel']) == 1]
data_DT_wh2 = data.loc[abs(data['wheel']) == 2]
data_DT_wh0_st123 = data.loc[(data['wheel'] == 0) & (data['station'] < 4)]
data_DT_wh1_st123 = data.loc[(abs(data['wheel']) == 1) & (data['station'] < 4)]
data_DT_wh2_st123 = data.loc[(abs(data['wheel']) == 2) & (data['station'] < 4)]
data_DT_wh0_st1 = data.loc[(data['wheel'] == 0) & (data['station'] == 1)]
data_DT_wh1_st1 = data.loc[(abs(data['wheel']) == 1) & (data['station'] == 1)]
data_DT_wh2_st1 = data.loc[(abs(data['wheel']) == 2) & (data['station'] == 1)]
data_DT_wh0_st2 = data.loc[(data['wheel'] == 0) & (data['station'] == 2)]
data_DT_wh1_st2 = data.loc[(abs(data['wheel']) == 1) & (data['station'] == 2)]
data_DT_wh2_st2 = data.loc[(abs(data['wheel']) == 2) & (data['station'] == 2)]
data_DT_wh0_st3 = data.loc[(data['wheel'] == 0) & (data['station'] == 3)]
data_DT_wh1_st3 = data.loc[(abs(data['wheel']) == 1) & (data['station'] == 3)]
data_DT_wh2_st3 = data.loc[(abs(data['wheel']) == 2) & (data['station'] == 3)]
data_DT_wh0_st4 = data.loc[(data['wheel'] == 0) & (data['station'] == 4)]
data_DT_wh1_st4 = data.loc[(abs(data['wheel']) == 1) & (data['station'] == 4)]
data_DT_wh2_st4 = data.loc[(abs(data['wheel']) == 2) & (data['station'] == 4)]
data_DT_st4_sec911 = data.loc[(data['station'] == 4) & ((data['sector'] == 9)
                                                        | (data['sector'] == 11))]
data_DT_st4_sec_sans911 = data.loc[(data['station'] == 4) & ((data['sector'] != 9) & (data['sector'] != 11))]

# Set up figure.
fig = plt.figure()
fig.suptitle("Drift Tube APEs by DOF (2018 Data)", fontsize=10)
fig.subplots_adjust(hspace=0.3, wspace=0.4)
gs = gridspec.GridSpec(2, 3)

# X
ax = fig.add_subplot(gs[0, 0])
plt.scatter(data_DT_st1["NSegments"], data_DT_st1["xx"], c='b', marker="o", label="Station 1")
plt.scatter(data_DT_st2["NSegments"], data_DT_st2["xx"], c='m', marker="o", label="Station 2")
plt.scatter(data_DT_st3["NSegments"], data_DT_st3["xx"], c='y', marker="o", label='Station 3')
plt.scatter(data_DT_st4["NSegments"], data_DT_st4["xx"], c='r', marker="o", label='Station 4')
optimizedParameters_x1, pcov_x1 = opt.curve_fit(uncert, data_DT_st1[
    "NSegments"], data_DT_st1["xx"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_x1), c='b')
optimizedParameters_x2, pcov_x2 = opt.curve_fit(uncert, data_DT_st2[
    "NSegments"], data_DT_st2["xx"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_x2), c='m')
optimizedParameters_x3, pcov_x3 = opt.curve_fit(uncert, data_DT_st3[
    "NSegments"], data_DT_st3["xx"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_x3), c='y')
optimizedParameters_x4, pcov_x4 = opt.curve_fit(uncert, data_DT_st4[
    "NSegments"], data_DT_st4["xx"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_x4), c='r')

print("X: Station 1 sigma_syst = ", optimizedParameters_x1[1])
print("X: Station 2 sigma_syst = ", optimizedParameters_x2[1])
print("X: Station 3 sigma_syst = ", optimizedParameters_x3[1])
print("X: Station 4 sigma_syst = ", optimizedParameters_x4[1])
print("X 68/95 Percentiles ---")
print("data_DT_st1", (data_DT_st1['xx']).quantile(0.68), (data_DT_st1[
    'xx']).quantile(0.95))
print("data_DT_st2", (data_DT_st2['xx']).quantile(0.68), (data_DT_st2[
    'xx']).quantile(0.95))
print("data_DT_st3", (data_DT_st3['xx']).quantile(0.68), (data_DT_st3[
    'xx']).quantile(0.95))
print("data_DT_st4", (data_DT_st4['xx']).quantile(0.68), (data_DT_st4[
    'xx']).quantile(0.95))

plt.semilogy()
plt.legend()
plt.title(r'$\sigma_{x}$', fontsize=10)
plt.ylabel("APE (micron)", fontsize=6)

# plt.savefig("xx_DT.png")
# plt.clf()


# Y
ax = fig.add_subplot(gs[0, 1])
plt.scatter(data_DT_wh0_st123["NSegments"],
            data_DT_wh0_st123["yy"],
            c='b', marker="o", label="Wheel 0")
plt.scatter(data_DT_wh1_st123["NSegments"],
            data_DT_wh1_st123["yy"],
            c='m', marker="o", label="Wheels +/- 1")
plt.scatter(data_DT_wh2_st123["NSegments"],
            data_DT_wh2_st123["yy"],
            c='c', marker="o", label="Wheels +/- 2")
optimizedParameters_y1, pcov_y1 = opt.curve_fit(uncert, data_DT_wh0_st123[
    "NSegments"], data_DT_wh0_st123["yy"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_y1), c='b')
optimizedParameters_y2, pcov_y2 = opt.curve_fit(uncert, data_DT_wh1_st123[
    "NSegments"], data_DT_wh1_st123["yy"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_y2), c='m')
optimizedParameters_y3, pcov_y3 = opt.curve_fit(uncert, data_DT_wh2_st123[
    "NSegments"], data_DT_wh2_st123["yy"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_y3), c='c')

print("Y: Wheel 0 sigma_syst = ", optimizedParameters_y1[1])
print("Y: Wheel 1 sigma_syst = ", optimizedParameters_y2[1])
print("Y: Wheel 2 sigma_syst = ", optimizedParameters_y3[1])
print("Y 68/95 Percentiles ---")
print("data_DT_st1", (data_DT_st1['yy']).quantile(0.68), (data_DT_st1[
    'yy']).quantile(0.95))
print("data_DT_st2", (data_DT_st2['yy']).quantile(0.68), (data_DT_st2[
    'yy']).quantile(0.95))
print("data_DT_st3", (data_DT_st3['yy']).quantile(0.68), (data_DT_st3[
    'yy']).quantile(0.95))

plt.semilogy()
plt.legend()
plt.title(r'$\sigma_{y}$', fontsize=10)

# plt.savefig("yy_DT.png")
# plt.clf()


# Z
ax = fig.add_subplot(gs[0, 2])
plt.scatter(data_DT_st12["NSegments"],
            data_DT_st12["zz"],
            c='b', marker="o", label="Stations 1 and 2")
plt.scatter(data_DT_st3["NSegments"],
            data_DT_st3["zz"],
            c='y', marker="o", label='Station 3')
plt.scatter(data_DT_st4_sec_sans911["NSegments"],
            data_DT_st4_sec_sans911["zz"],
            c='r', marker="o", label='Station 4 sans Sectors 9,11')
plt.scatter(data_DT_st4_sec911["NSegments"],
            data_DT_st4_sec911["zz"],
            c='r', marker="+", label='Station 4 Sectors 9,11')

optimizedParameters_z1, pcov_z1 = opt.curve_fit(uncert, data_DT_st12[
    "NSegments"], data_DT_st12["zz"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_z1), c='b')
optimizedParameters_z2, pcov_z2 = opt.curve_fit(uncert, data_DT_st3[
    "NSegments"], data_DT_st3["zz"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_z2), c='y')
optimizedParameters_z3, pcov_z3 = opt.curve_fit(uncert, data_DT_st4_sec_sans911[
    "NSegments"], data_DT_st4_sec_sans911["zz"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_z3), c='r')
optimizedParameters_z4, pcov_z4 = opt.curve_fit(uncert, data_DT_st4_sec911[
    "NSegments"], data_DT_st4_sec911["zz"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_z4), c='r', ls='dashed')

print("Z: Stations 1&2 sigma_syst = ", optimizedParameters_z1[1])
print("Z: Station 3 sigma_syst = ", optimizedParameters_z2[1])
print("Z: Station 4 (sans sectors 9,11) sigma_syst = ",
      optimizedParameters_z3[1])
print("Z: Station 4 Sectors 9,11 sigma_syst = ", optimizedParameters_z4[1])
print("Z 68/95 Percentiles ---")
print("data_DT_st12", (data_DT_st12['zz']).quantile(0.68), (data_DT_st12[
    'zz']).quantile(0.95))
print("data_DT_st3", (data_DT_st3['zz']).quantile(0.68), (data_DT_st3[
    'zz']).quantile(0.95))
print("data_DT_st4", (data_DT_st4['zz']).quantile(0.68), (data_DT_st4[
    'zz']).quantile(0.95))

plt.semilogy()
plt.legend()
plt.title(r'$\sigma_{z}$', fontsize=10)

# plt.savefig("zz_DT.png")
# plt.clf()


# PHI X
ax = fig.add_subplot(gs[1, 0])
# AT: probably need to split by sector
plt.scatter(data_DT_wh0_st1["NSegments"],
            data_DT_wh0_st1["aa"],
            c='b', marker="o", label="St1, Wheel 0")
plt.scatter(data_DT_wh0_st2["NSegments"],
            data_DT_wh0_st2["aa"],
            c='b', marker="d", label="St2, Wheel 0")
plt.scatter(data_DT_wh0_st3["NSegments"],
            data_DT_wh0_st3["aa"],
            c='b', marker="+", label="St3, Wheel 0")

plt.scatter(data_DT_wh1_st1["NSegments"],
            data_DT_wh1_st1["aa"],
            c='m', marker="o", label="St1, Wheel 1")
plt.scatter(data_DT_wh1_st2["NSegments"],
            data_DT_wh1_st2["aa"],
            c='m', marker="d", label="St2, Wheel 1")
plt.scatter(data_DT_wh1_st3["NSegments"],
            data_DT_wh1_st3["aa"],
            c='m', marker="+", label="St3, Wheel 1")

plt.scatter(data_DT_wh2_st1["NSegments"],
            data_DT_wh2_st1["aa"],
            c='c', marker="o", label="St1, Wheel 2")
plt.scatter(data_DT_wh2_st2["NSegments"],
            data_DT_wh2_st2["aa"],
            c='c', marker="d", label="St2, Wheel 2")
plt.scatter(data_DT_wh2_st3["NSegments"],
            data_DT_wh2_st3["aa"],
            c='c', marker="+", label="St3, Wheel 2")

plt.scatter(data_DT_st4_sec_sans911["NSegments"],
            data_DT_st4_sec_sans911["aa"],
            c='r', marker="o", label='Station 4 (no Sectors 9,11)')
plt.scatter(data_DT_st4_sec911["NSegments"],
            data_DT_st4_sec911["aa"],
            c='r', marker="+", label='Station 4, Sectors 9,11')

optimizedParameters_phix11, pcov_phix11 = opt.curve_fit(uncert, data_DT_wh0_st1[
    "NSegments"], data_DT_wh0_st1["aa"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phix11),
         c='b')
optimizedParameters_phix12, pcov_phix12 = opt.curve_fit(uncert, data_DT_wh0_st2[
    "NSegments"], data_DT_wh0_st2["aa"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phix12),
         c='b', ls='dashed')
optimizedParameters_phix13, pcov_phix13 = opt.curve_fit(uncert, data_DT_wh0_st3[
    "NSegments"], data_DT_wh0_st3["aa"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phix13),
         c='b', ls='dotted')

optimizedParameters_phix21, pcov_phix21 = opt.curve_fit(uncert, data_DT_wh1_st1[
    "NSegments"], data_DT_wh1_st1["aa"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phix21),
         c='m')
optimizedParameters_phix22, pcov_phix22 = opt.curve_fit(uncert, data_DT_wh1_st2[
    "NSegments"], data_DT_wh1_st2["aa"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phix22),
         c='m', ls='dashed')
optimizedParameters_phix23, pcov_phix23 = opt.curve_fit(uncert, data_DT_wh1_st3[
    "NSegments"], data_DT_wh1_st3["aa"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phix23),
         c='m', ls='dotted')

optimizedParameters_phix31, pcov_phix31 = opt.curve_fit(uncert, data_DT_wh2_st1[
    "NSegments"], data_DT_wh2_st1["aa"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phix31),
         c='c')
optimizedParameters_phix32, pcov_phix32 = opt.curve_fit(uncert, data_DT_wh2_st2[
    "NSegments"], data_DT_wh2_st2["aa"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phix32),
         c='c', ls='dashed')
optimizedParameters_phix33, pcov_phix33 = opt.curve_fit(uncert, data_DT_wh2_st3[
    "NSegments"], data_DT_wh2_st3["aa"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phix33),
         c='c', ls='dotted')

optimizedParameters_phix4, pcov_phix4 = opt.curve_fit(uncert, data_DT_st4_sec_sans911["NSegments"],
                                                      data_DT_st4_sec_sans911["aa"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phix4), c='r')
optimizedParameters_phix5, pcov_phix5 = opt.curve_fit(uncert, data_DT_st4_sec911["NSegments"], data_DT_st4_sec911["aa"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phix5), ls='dashed', c='r')

print("PHI X: MB 0/1 sigma_syst = ", optimizedParameters_phix11[1])
print("PHI X: MB 0/2 sigma_syst = ", optimizedParameters_phix12[1])
print("PHI X: MB 0/3 sigma_syst = ", optimizedParameters_phix13[1])
print("PHI X: MB +/- 1/1 sigma_syst = ", optimizedParameters_phix21[1])
print("PHI X: MB +/- 1/2 sigma_syst = ", optimizedParameters_phix22[1])
print("PHI X: MB +/- 1/3 sigma_syst = ", optimizedParameters_phix23[1])
print("PHI X: MB +/- 2/1 sigma_syst = ", optimizedParameters_phix31[1])
print("PHI X: MB +/- 2/2 sigma_syst = ", optimizedParameters_phix32[1])
print("PHI X: MB +/- 2/3 sigma_syst = ", optimizedParameters_phix33[1])
print("PHI X: Station 4 (sans sectors 9,11) sigma_syst = ", optimizedParameters_phix4[1])
print("PHI X: Station 4 Sectors 9,11 sigma_syst = ", optimizedParameters_phix5[1])

print("PHI X 68/95 Percentiles ---")
print("data_DT_wh0_st1", (data_DT_wh0_st1['aa']).quantile(0.68), (data_DT_wh0_st1['aa']).quantile(0.95))
print("data_DT_wh0_st2", (data_DT_wh0_st2['aa']).quantile(0.68), (data_DT_wh0_st2['aa']).quantile(0.95))
print("data_DT_wh0_st3", (data_DT_wh0_st3['aa']).quantile(0.68), (data_DT_wh0_st3['aa']).quantile(0.95))
print("data_DT_wh1_st1", (data_DT_wh1_st1['aa']).quantile(0.68), (data_DT_wh1_st1['aa']).quantile(0.95))
print("data_DT_wh1_st2", (data_DT_wh1_st2['aa']).quantile(0.68), (data_DT_wh1_st2['aa']).quantile(0.95))
print("data_DT_wh1_st3", (data_DT_wh1_st3['aa']).quantile(0.68), (data_DT_wh1_st3['aa']).quantile(0.95))
print("data_DT_wh2_st1", (data_DT_wh2_st1['aa']).quantile(0.68), (data_DT_wh2_st1['aa']).quantile(0.95))
print("data_DT_wh2_st2", (data_DT_wh2_st2['aa']).quantile(0.68), (data_DT_wh2_st2['aa']).quantile(0.95))
print("data_DT_wh2_st3", (data_DT_wh2_st3['aa']).quantile(0.68), (data_DT_wh2_st3['aa']).quantile(0.95))
print("data_DT_st4_sec_sans911", (data_DT_st4_sec_sans911['aa']).quantile(0.68),
      (data_DT_st4_sec_sans911['aa']).quantile(0.95))
print("data_DT_st4_sec911", (data_DT_st4_sec911['aa']).quantile(0.68), (data_DT_st4_sec911['aa']).quantile(0.95))

plt.semilogy()
plt.legend(ncol=2, fontsize=3)
plt.title(r'$\sigma_{\phi_x}$', fontsize=10)
plt.ylabel("APE (mrad)", fontsize=6)

# plt.savefig("aa_DT.png")
# plt.clf()


# PHI Y
ax = fig.add_subplot(gs[1, 1])
plt.scatter(data_DT_wh0_st1["NSegments"],
            data_DT_wh0_st1["bb"],
            c='b', marker="o", label="St1, Wheel 0")
plt.scatter(data_DT_wh0_st2["NSegments"],
            data_DT_wh0_st2["bb"],
            c='b', marker="d", label="St2, Wheel 0")
plt.scatter(data_DT_wh0_st3["NSegments"],
            data_DT_wh0_st3["bb"],
            c='b', marker="+", label="St3, Wheel 0")

plt.scatter(data_DT_wh1_st1["NSegments"],
            data_DT_wh1_st1["bb"],
            c='m', marker="o", label="St1, Wheel 1")
plt.scatter(data_DT_wh1_st2["NSegments"],
            data_DT_wh1_st2["bb"],
            c='m', marker="d", label="St2, Wheel 1")
plt.scatter(data_DT_wh1_st3["NSegments"],
            data_DT_wh1_st3["bb"],
            c='m', marker="+", label="St3, Wheel 1")

plt.scatter(data_DT_wh2_st1["NSegments"],
            data_DT_wh2_st1["bb"],
            c='c', marker="o", label="St1, Wheel 2")
plt.scatter(data_DT_wh2_st2["NSegments"],
            data_DT_wh2_st2["bb"],
            c='c', marker="d", label="St2, Wheel 2")
plt.scatter(data_DT_wh2_st3["NSegments"],
            data_DT_wh2_st3["bb"],
            c='c', marker="+", label="St3, Wheel 2")

plt.scatter(data_DT_st4_sec_sans911["NSegments"],
            data_DT_st4_sec_sans911["bb"],
            c='r', marker="o", label='Station 4 (no Sectors 9,11)')
plt.scatter(data_DT_st4_sec911["NSegments"],
            data_DT_st4_sec911["bb"],
            c='r', marker="+", label='Station 4, Sectors 9,11')

optimizedParameters_phiy11, pcov_phiy11 = opt.curve_fit(uncert, data_DT_wh0_st1[
    "NSegments"], data_DT_wh0_st1["bb"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phiy11),
         c='b')
optimizedParameters_phiy12, pcov_phiy12 = opt.curve_fit(uncert, data_DT_wh0_st2[
    "NSegments"], data_DT_wh0_st2["bb"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phiy12),
         c='b', ls='dashed')
optimizedParameters_phiy13, pcov_phiy13 = opt.curve_fit(uncert, data_DT_wh0_st3[
    "NSegments"], data_DT_wh0_st3["bb"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phiy13),
         c='b', ls='dotted')

optimizedParameters_phiy21, pcov_phiy21 = opt.curve_fit(uncert, data_DT_wh1_st1[
    "NSegments"], data_DT_wh1_st1["bb"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phiy21),
         c='m')
optimizedParameters_phiy22, pcov_phiy22 = opt.curve_fit(uncert, data_DT_wh1_st2[
    "NSegments"], data_DT_wh1_st2["bb"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phiy22),
         c='m', ls='dashed')
optimizedParameters_phiy23, pcov_phiy23 = opt.curve_fit(uncert, data_DT_wh1_st3[
    "NSegments"], data_DT_wh1_st3["bb"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phiy23),
         c='m', ls='dotted')

optimizedParameters_phiy31, pcov_phiy31 = opt.curve_fit(uncert, data_DT_wh2_st1[
    "NSegments"], data_DT_wh2_st1["bb"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phiy31),
         c='c')
optimizedParameters_phiy32, pcov_phiy32 = opt.curve_fit(uncert, data_DT_wh2_st2[
    "NSegments"], data_DT_wh2_st2["bb"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phiy32),
         c='c', ls='dashed')
optimizedParameters_phiy33, pcov_phiy33 = opt.curve_fit(uncert, data_DT_wh2_st3[
    "NSegments"], data_DT_wh2_st3["bb"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phiy33),
         c='c', ls='dotted')

optimizedParameters_phiy4, pcov_phiy4 = opt.curve_fit(uncert,
                                                      data_DT_st4_sec_sans911["NSegments"],
                                                      data_DT_st4_sec_sans911["bb"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phiy4), c='r')
optimizedParameters_phiy5, pcov_phiy5 = opt.curve_fit(uncert,
                                                      data_DT_st4_sec911["NSegments"], data_DT_st4_sec911["bb"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phiy5), ls='dashed', c='r')

print("PHI Y: MB 0/1 sigma_syst = ", optimizedParameters_phiy11[1])
print("PHI Y: MB 0/2 sigma_syst = ", optimizedParameters_phiy12[1])
print("PHI Y: MB 0/3 sigma_syst = ", optimizedParameters_phiy13[1])
print("PHI Y: MB +/- 1/1 sigma_syst = ", optimizedParameters_phiy21[1])
print("PHI Y: MB +/- 1/2 sigma_syst = ", optimizedParameters_phiy22[1])
print("PHI Y: MB +/- 1/3 sigma_syst = ", optimizedParameters_phiy23[1])
print("PHI Y: MB +/- 2/1 sigma_syst = ", optimizedParameters_phiy31[1])
print("PHI Y: MB +/- 2/2 sigma_syst = ", optimizedParameters_phiy32[1])
print("PHI Y: MB +/- 2/3 sigma_syst = ", optimizedParameters_phiy33[1])
print("PHI Y: Station 4 (sans sectors 9,11) sigma_syst = ", optimizedParameters_phiy4[1])
print("PHI Y: Station 4 Sectors 9,11 sigma_syst = ", optimizedParameters_phiy5[1])
print("PHI Y 68/95 Percentiles ---")
print("data_DT_wh0_st1", (data_DT_wh0_st1['bb']).quantile(0.68), (data_DT_wh0_st1['bb']).quantile(0.95))
print("data_DT_wh0_st2", (data_DT_wh0_st2['bb']).quantile(0.68), (data_DT_wh0_st2['bb']).quantile(0.95))
print("data_DT_wh0_st3", (data_DT_wh0_st3['bb']).quantile(0.68), (data_DT_wh0_st3['bb']).quantile(0.95))
print("data_DT_wh1_st1", (data_DT_wh1_st1['bb']).quantile(0.68), (data_DT_wh1_st1['bb']).quantile(0.95))
print("data_DT_wh1_st2", (data_DT_wh1_st2['bb']).quantile(0.68), (data_DT_wh1_st2['bb']).quantile(0.95))
print("data_DT_wh1_st3", (data_DT_wh1_st3['bb']).quantile(0.68), (data_DT_wh1_st3['bb']).quantile(0.95))
print("data_DT_wh2_st1", (data_DT_wh2_st1['bb']).quantile(0.68), (data_DT_wh2_st1['bb']).quantile(0.95))
print("data_DT_wh2_st2", (data_DT_wh2_st2['bb']).quantile(0.68), (data_DT_wh2_st2['bb']).quantile(0.95))
print("data_DT_wh2_st3", (data_DT_wh2_st3['bb']).quantile(0.68), (data_DT_wh2_st3['bb']).quantile(0.95))
print("data_DT_st4_sec_sans911", (data_DT_st4_sec_sans911['bb']).quantile(0.68),
      (data_DT_st4_sec_sans911['bb']).quantile(0.95))
print("data_DT_st4_sec911", (data_DT_st4_sec911['bb']).quantile(0.68), (data_DT_st4_sec911['bb']).quantile(0.95))


plt.semilogy()
plt.legend(ncol=2, fontsize=3)
plt.title(r'$\sigma_{\phi_y}$', fontsize=10)
plt.xlabel("1000's of Segments", fontsize=6)

# plt.savefig("bb_DT.png")
# plt.clf()


# PHI Z
ax = fig.add_subplot(gs[1, 2])
plt.scatter(data_DT_wh0_st1["NSegments"],
            data_DT_wh0_st1["cc"],
            c='b', marker="o", label="St1, Wheel 0")
plt.scatter(data_DT_wh0_st2["NSegments"],
            data_DT_wh0_st2["cc"],
            c='b', marker="d", label="St2, Wheel 0")
plt.scatter(data_DT_wh0_st3["NSegments"],
            data_DT_wh0_st3["cc"],
            c='b', marker="+", label="St3, Wheel 0")

plt.scatter(data_DT_wh1_st1["NSegments"],
            data_DT_wh1_st1["cc"],
            c='m', marker="o", label="St1, Wheel 1")
plt.scatter(data_DT_wh1_st2["NSegments"],
            data_DT_wh1_st2["cc"],
            c='m', marker="d", label="St2, Wheel 1")
plt.scatter(data_DT_wh1_st3["NSegments"],
            data_DT_wh1_st3["cc"],
            c='m', marker="+", label="St3, Wheel 1")

plt.scatter(data_DT_wh2_st1["NSegments"],
            data_DT_wh2_st1["cc"],
            c='c', marker="o", label="St1, Wheel 2")
plt.scatter(data_DT_wh2_st2["NSegments"],
            data_DT_wh2_st2["cc"],
            c='c', marker="d", label="St2, Wheel 2")
plt.scatter(data_DT_wh2_st3["NSegments"],
            data_DT_wh2_st3["cc"],
            c='c', marker="+", label="St3, Wheel 2")

plt.scatter(data_DT_st4_sec_sans911["NSegments"],
            data_DT_st4_sec_sans911["cc"],
            c='r', marker="o", label='Station 4 (no Sectors 9,11)')
plt.scatter(data_DT_st4_sec911["NSegments"],
            data_DT_st4_sec911["cc"],
            c='r', marker="+", label='Station 4, Sectors 9,11')

optimizedParameters_phiz11, pcov_phiz11 = opt.curve_fit(uncert, data_DT_wh0_st1[
    "NSegments"], data_DT_wh0_st1["cc"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phiz11),
         c='b')
optimizedParameters_phiz12, pcov_phiz12 = opt.curve_fit(uncert, data_DT_wh0_st2[
    "NSegments"], data_DT_wh0_st2["cc"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phiz12),
         c='b', ls='dashed')
optimizedParameters_phiz13, pcov_phiz13 = opt.curve_fit(uncert, data_DT_wh0_st3[
    "NSegments"], data_DT_wh0_st3["cc"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phiz13),
         c='b', ls='dotted')

optimizedParameters_phiz21, pcov_phiz21 = opt.curve_fit(uncert, data_DT_wh1_st1[
    "NSegments"], data_DT_wh1_st1["cc"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phiz21),
         c='m')
optimizedParameters_phiz22, pcov_phiz22 = opt.curve_fit(uncert, data_DT_wh1_st2[
    "NSegments"], data_DT_wh1_st2["cc"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phiz22),
         c='m', ls='dashed')
optimizedParameters_phiz23, pcov_phiz23 = opt.curve_fit(uncert, data_DT_wh1_st3[
    "NSegments"], data_DT_wh1_st3["cc"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phiz23),
         c='m', ls='dotted')

optimizedParameters_phiz31, pcov_phiz31 = opt.curve_fit(uncert, data_DT_wh2_st1[
    "NSegments"], data_DT_wh2_st1["cc"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phiz31),
         c='c')
optimizedParameters_phiz32, pcov_phiz32 = opt.curve_fit(uncert, data_DT_wh2_st2[
    "NSegments"], data_DT_wh2_st2["cc"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phiz32),
         c='c', ls='dashed')
optimizedParameters_phiz33, pcov_phiz33 = opt.curve_fit(uncert, data_DT_wh2_st3[
    "NSegments"], data_DT_wh2_st3["cc"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phiz33),
         c='c', ls='dotted')

optimizedParameters_phiz4, pcov_phiz4 = opt.curve_fit(uncert,
                                                      data_DT_st4_sec_sans911["NSegments"],
                                                      data_DT_st4_sec_sans911["cc"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phiz4), c='r')
optimizedParameters_phiz5, pcov_phiz5 = opt.curve_fit(uncert,
                                                      data_DT_st4_sec911["NSegments"], data_DT_st4_sec911["cc"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_phiz5), ls='dashed', c='r')


print("PHI Z: MB 0/1 sigma_syst = ", optimizedParameters_phiz11[1])
print("PHI Z: MB 0/2 sigma_syst = ", optimizedParameters_phiz12[1])
print("PHI Z: MB 0/3 sigma_syst = ", optimizedParameters_phiz13[1])
print("PHI Z: MB +/- 1/1 sigma_syst = ", optimizedParameters_phiz21[1])
print("PHI Z: MB +/- 1/2 sigma_syst = ", optimizedParameters_phiz22[1])
print("PHI Z: MB +/- 1/3 sigma_syst = ", optimizedParameters_phiz23[1])
print("PHI Z: MB +/- 2/1 sigma_syst = ", optimizedParameters_phiz31[1])
print("PHI Z: MB +/- 2/2 sigma_syst = ", optimizedParameters_phiz32[1])
print("PHI Z: MB +/- 2/3 sigma_syst = ", optimizedParameters_phiz33[1])
print("PHI Z: Station 4 (sans sectors 9,11) sigma_syst = ", optimizedParameters_phiz4[1])
print("PHI Z: Station 4 Sectors 9,11 sigma_syst = ", optimizedParameters_phiz5[1])
print("PHI Z 68/95 Percentiles ---")
print("data_DT_wh0_st1", (data_DT_wh0_st1['cc']).quantile(0.68), (data_DT_wh0_st1['cc']).quantile(0.95))
print("data_DT_wh0_st2", (data_DT_wh0_st2['cc']).quantile(0.68), (data_DT_wh0_st2['cc']).quantile(0.95))
print("data_DT_wh0_st3", (data_DT_wh0_st3['cc']).quantile(0.68), (data_DT_wh0_st3['cc']).quantile(0.95))
print("data_DT_wh1_st1", (data_DT_wh1_st1['cc']).quantile(0.68), (data_DT_wh1_st1['cc']).quantile(0.95))
print("data_DT_wh1_st2", (data_DT_wh1_st2['cc']).quantile(0.68), (data_DT_wh1_st2['cc']).quantile(0.95))
print("data_DT_wh1_st3", (data_DT_wh1_st3['cc']).quantile(0.68), (data_DT_wh1_st3['cc']).quantile(0.95))
print("data_DT_wh2_st1", (data_DT_wh2_st1['cc']).quantile(0.68), (data_DT_wh2_st1['cc']).quantile(0.95))
print("data_DT_wh2_st2", (data_DT_wh2_st2['cc']).quantile(0.68), (data_DT_wh2_st2['cc']).quantile(0.95))
print("data_DT_wh2_st3", (data_DT_wh2_st3['cc']).quantile(0.68), (data_DT_wh2_st3['cc']).quantile(0.95))
print("data_DT_st4_sec_sans911", (data_DT_st4_sec_sans911['cc']).quantile(0.68),
      (data_DT_st4_sec_sans911['cc']).quantile(0.95))
print("data_DT_st4_sec911", (data_DT_st4_sec911['cc']).quantile(0.68), (data_DT_st4_sec911['cc']).quantile(0.95))

plt.semilogy()
plt.legend(ncol=2, fontsize=3)
plt.title(r'$\sigma_{\phi_x}$', fontsize=10)


# plt.savefig("cc_DT.png")

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig("uncertainty_curves_allDOF.png")
plt.savefig("uncertainty_curves_allDOF.pdf")
