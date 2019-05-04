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
mpl.rcParams['font.size'] = 5
mpl.rcParams['lines.markersize'] = 2

data = pd.read_csv("covariance_csc_iov12.csv")

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
data_CSC_ME1 = data.loc[data['endcap'] == 1]
data_CSC_ME2 = data.loc[data['endcap'] == 2]
data_CSC_st1 = data.loc[data['station'] == 1]
data_CSC_st2 = data.loc[data['station'] == 2]
data_CSC_st3 = data.loc[data['station'] == 3]  # station 4 is identical to 1
data_CSC_ring1 = data.loc[data['ring'] == 1]
data_CSC_ring2 = data.loc[data['ring'] == 2]
data_CSC_ring3 = data.loc[data['ring'] == 3]
data_CSC_ring1st1 = data.loc[(data['ring'] == 1) & (data['station'] == 1)]
data_CSC_ring1st23 = data.loc[(data['ring'] == 1) & (data['station'] > 1)]
data_CSC_ring2st1 = data.loc[(data['ring'] == 2) & (data['station'] == 1)]
data_CSC_ring2st23 = data.loc[(data['ring'] == 2) & (data['station'] > 1)]


# Set up figure.
fig = plt.figure(figsize=(8, 4))
fig.suptitle("Cathode Strip Chamber APEs by DOF (2018 Data)", fontsize=10)
fig.subplots_adjust(hspace=0.3, wspace=0.4)
gs = gridspec.GridSpec(1, 3)

# X
ax = fig.add_subplot(gs[0, 0])
plt.scatter(data_CSC_ring1st1["NSegments"], data_CSC_ring1st1["xx"], c='b', marker="o", label="ME$\pm$1/1")
plt.scatter(data_CSC_ring1st23["NSegments"], data_CSC_ring1st23["xx"], c='b', marker="d", label="ME$\pm$2/1 & "
                                                                                                "ME$\pm$3/1")
plt.scatter(data_CSC_ring2st1["NSegments"], data_CSC_ring2st1["xx"], c='m', marker="o", label="ME$\pm$1/2")
plt.scatter(data_CSC_ring2st23["NSegments"], data_CSC_ring2st23["xx"], c='m', marker="d", label="ME$\pm$2/2 & "
                                                                                                "ME$\pm$3/2")
plt.scatter(data_CSC_ring3["NSegments"], data_CSC_ring3["xx"], c='y', marker="o", label='ME$\pm$*/3')
optimizedParameters_x1, pcov_x1 = opt.curve_fit(uncert, data_CSC_ring1st1[
    "NSegments"], data_CSC_ring1st1["xx"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_x1), c='b')
optimizedParameters_x2, pcov_x2 = opt.curve_fit(uncert, data_CSC_ring1st23[
    "NSegments"], data_CSC_ring1st23["xx"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_x2), c='b', ls = 'dashed')
optimizedParameters_x3, pcov_x3 = opt.curve_fit(uncert, data_CSC_ring2st1[
    "NSegments"], data_CSC_ring2st1["xx"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_x3), c='m')
optimizedParameters_x4, pcov_x4 = opt.curve_fit(uncert, data_CSC_ring2st23[
    "NSegments"], data_CSC_ring2st23["xx"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_x4), c='m', ls = 'dashed')
optimizedParameters_x5, pcov_x5 = opt.curve_fit(uncert, data_CSC_ring3[
    "NSegments"], data_CSC_ring3["xx"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_x5), c='y')


print("X: Ring 1 Station 1 sigma_syst = ", optimizedParameters_x1[1])
print("X: Ring 1 Station 2,3 sigma_syst = ", optimizedParameters_x2[1])
print("X: Ring 2 sigma_syst = ", optimizedParameters_x3[1])
print("X: Ring 2 Station 2,3 sigma_syst = ", optimizedParameters_x4[1])
print("X: Ring 3 sigma_syst = ", optimizedParameters_x5[1])
print("X 68/95 Percentiles ---")
print("data_CSC_ring1st1", (data_CSC_ring1st1['xx']).quantile(0.68), (data_CSC_ring1st1['xx']).quantile(0.95))
print("data_CSC_ring1st23", (data_CSC_ring1st23['xx']).quantile(0.68), (data_CSC_ring1st23['xx']).quantile(0.95))
print("data_CSC_ring2st1", (data_CSC_ring2st1['xx']).quantile(0.68), (data_CSC_ring2st1['xx']).quantile(0.95))
print("data_CSC_ring2st23", (data_CSC_ring2st23['xx']).quantile(0.68), (data_CSC_ring2st23['xx']).quantile(0.95))
print("data_CSC_ring3", (data_CSC_st3['xx']).quantile(0.68), (data_CSC_st3['xx']).quantile(0.95))


plt.semilogy()
plt.legend()
plt.title(r'$\sigma_{x}$', fontsize=10)
plt.ylabel("APE (micron)", fontsize=6)



# Y
ax = fig.add_subplot(gs[0, 1])
plt.scatter(data_CSC_ring1st1["NSegments"], data_CSC_ring1st1["yy"], c='b', marker="o", label="ME$\pm$1/1")
plt.scatter(data_CSC_ring1st23["NSegments"], data_CSC_ring1st23["yy"], c='b', marker="d", label="ME$\pm$2/1 & "
                                                                                                "ME$\pm$3/1")
plt.scatter(data_CSC_ring2st1["NSegments"], data_CSC_ring2st1["yy"], c='m', marker="o", label="ME$\pm$1/2")
plt.scatter(data_CSC_ring2st23["NSegments"], data_CSC_ring2st23["yy"], c='m', marker="d", label="ME$\pm$2/2 & "
                                                                                                "ME$\pm$3/2")
plt.scatter(data_CSC_ring3["NSegments"], data_CSC_ring3["yy"], c='y', marker="o", label='ME$\pm$*/3')
optimizedParameters_x1, pcov_x1 = opt.curve_fit(uncert, data_CSC_ring1st1[
    "NSegments"], data_CSC_ring1st1["yy"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_x1), c='b')
optimizedParameters_x2, pcov_x2 = opt.curve_fit(uncert, data_CSC_ring1st23[
    "NSegments"], data_CSC_ring1st23["yy"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_x2), c='b', ls = 'dashed')
optimizedParameters_x3, pcov_x3 = opt.curve_fit(uncert, data_CSC_ring2st1[
    "NSegments"], data_CSC_ring2st1["yy"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_x3), c='m')
optimizedParameters_x4, pcov_x4 = opt.curve_fit(uncert, data_CSC_ring2st23[
    "NSegments"], data_CSC_ring2st23["yy"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_x4), c='m', ls = 'dashed')
optimizedParameters_x5, pcov_x5 = opt.curve_fit(uncert, data_CSC_ring3[
    "NSegments"], data_CSC_ring3["yy"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_x5), c='y')


print("Y: Ring 1 Station 1 sigma_syst = ", optimizedParameters_x1[1])
print("Y: Ring 1 Station 2,3 sigma_syst = ", optimizedParameters_x2[1])
print("Y: Ring 2 sigma_syst = ", optimizedParameters_x3[1])
print("Y: Ring 2 Station 2,3 sigma_syst = ", optimizedParameters_x4[1])
print("Y: Ring 3 sigma_syst = ", optimizedParameters_x5[1])
print("Y 68/95 Percentiles ---")
print("data_CSC_ring1st1", (data_CSC_ring1st1['yy']).quantile(0.68), (data_CSC_ring1st1['yy']).quantile(0.95))
print("data_CSC_ring1st23", (data_CSC_ring1st23['yy']).quantile(0.68), (data_CSC_ring1st23['yy']).quantile(0.95))
print("data_CSC_ring2st1", (data_CSC_ring2st1['yy']).quantile(0.68), (data_CSC_ring2st1['yy']).quantile(0.95))
print("data_CSC_ring2st23", (data_CSC_ring2st23['yy']).quantile(0.68), (data_CSC_ring2st23['yy']).quantile(0.95))
print("data_CSC_ring3", (data_CSC_st3['yy']).quantile(0.68), (data_CSC_st3['yy']).quantile(0.95))

plt.semilogy()
plt.legend()
plt.title(r'$\sigma_{y}$', fontsize=10)
plt.xlabel("1000s of Segments", fontsize=6)


# PHI Z
ax = fig.add_subplot(gs[0, 2])
plt.scatter(data_CSC_ring1st1["NSegments"], data_CSC_ring1st1["cc"], c='b', marker="o", label="ME$\pm$1/1")
plt.scatter(data_CSC_ring1st2["NSegments"], data_CSC_ring1st2["cc"], c='b', marker="x", label="ME$\pm$2/1")
plt.scatter(data_CSC_ring1st3["NSegments"], data_CSC_ring1st3["cc"], c='b', marker="+", label="ME$\pm$3/1")
plt.scatter(data_CSC_ring2st1["NSegments"], data_CSC_ring2st1["cc"], c='m', marker="o", label="ME$\pm$1/2")
plt.scatter(data_CSC_ring2st23["NSegments"], data_CSC_ring2st23["cc"], c='m', marker="d", label="ME$\pm$2/2 & "
                                                                                                "ME$\pm$3/2")
plt.scatter(data_CSC_ring3["NSegments"], data_CSC_ring3["cc"], c='y', marker="o", label='ME$\pm$*/3')
optimizedParameters_x1, pcov_x1 = opt.curve_fit(uncert, data_CSC_ring1st1[
    "NSegments"], data_CSC_ring1st1["cc"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_x1), c='b')
optimizedParameters_x2, pcov_x2 = opt.curve_fit(uncert, data_CSC_ring1st2[
    "NSegments"], data_CSC_ring1st2["cc"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_x2), c='b', ls = 'dashed')
optimizedParameters_x3, pcov_x3 = opt.curve_fit(uncert, data_CSC_ring1st3[
    "NSegments"], data_CSC_ring1st3["cc"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_x3), c='b', ls = 'dotted')
optimizedParameters_x4, pcov_x4 = opt.curve_fit(uncert, data_CSC_ring2st1[
    "NSegments"], data_CSC_ring2st1["cc"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_x4), c='m')
optimizedParameters_x5, pcov_x5 = opt.curve_fit(uncert, data_CSC_ring2st23[
    "NSegments"], data_CSC_ring2st23["cc"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_x5), c='m', ls = 'dashed')
optimizedParameters_x6, pcov_x6 = opt.curve_fit(uncert, data_CSC_ring3[
    "NSegments"], data_CSC_ring3["cc"])
plt.plot(n_seg, uncert(n_seg, *optimizedParameters_x6), c='y')


print("phiz: Ring 1 Station 1 sigma_syst = ", optimizedParameters_x1[1])
print("phiz: Ring 1 Station 2,3 sigma_syst = ", optimizedParameters_x2[1])
print("phiz: Ring 2 sigma_syst = ", optimizedParameters_x3[1])
print("phiz: Ring 2 Station 2,3 sigma_syst = ", optimizedParameters_x4[1])
print("phiz: Ring 3 sigma_syst = ", optimizedParameters_x5[1])
print("phiz 68/95 Percentiles ---")
print("data_CSC_ring1st1", (data_CSC_ring1st1['cc']).quantile(0.68), (data_CSC_ring1st1['cc']).quantile(0.95))
print("data_CSC_ring1st2", (data_CSC_ring1st2['cc']).quantile(0.68), (data_CSC_ring1st2['cc']).quantile(0.95))
print("data_CSC_ring1st3", (data_CSC_ring1st3['cc']).quantile(0.68), (data_CSC_ring1st3['cc']).quantile(0.95))
print("data_CSC_ring2st1", (data_CSC_ring2st1['cc']).quantile(0.68), (data_CSC_ring2st1['cc']).quantile(0.95))
print("data_CSC_ring2st23", (data_CSC_ring2st23['cc']).quantile(0.68), (data_CSC_ring2st23['cc']).quantile(0.95))
print("data_CSC_ring3", (data_CSC_st3['cc']).quantile(0.68), (data_CSC_st3['cc']).quantile(0.95))


plt.semilogy()
plt.legend()
plt.title(r'$\sigma_{\phi_z}$', fontsize=10)
plt.ylabel("APE (mrad)", fontsize=6)


plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig("uncertainty_curves_CSC_allDOF.png")
plt.savefig("uncertainty_curves_CSC_allDOF.pdf")
