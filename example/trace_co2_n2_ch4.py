# Test tern teqp
import teqp
import pandas as pd
import numpy as np
from ternary_teqp import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import mpltern


# Ternary VLE - Testing
# Thermodynamics of the carbon dioxide plus nitrogen plus methane (CO2 + N2 + CH4) system: 
# Measurements of vapor-liquid equilibrium data at temperatures from 223 to 298 K and verification of EOS-CG-2019 equation of state
# https://doi.org/10.1016/j.fluid.2019.112444

# Components 
names = ["CarbonDioxide","Nitrogen","Methane"]

pures = []; anc= [];

# Create pure fluids and ancillaries
for n in names:
    pures.append(teqp.build_multifluid_model([n], teqp.get_datapath()))
    anc.append(pures[-1].build_ancillaries())

# Create binary mixture (co2 - n2) and ternary mixture (co2 - n2 - ch4)
bin_mix  =teqp.build_multifluid_model(names[0:2], teqp.get_datapath())
tern_mix =teqp.build_multifluid_model(names, teqp.get_datapath())

TS = [220.0,273.15]  # / K 
PS =  [12E6 ,8.2E6 ]    # /  Pa

# Create figure
fig = plt.figure(figsize=(10.8, 8))
fig.subplots_adjust(wspace=0.3)

for idx,T in enumerate(TS):
    p = PS[idx]   
    # Trace binary mixture co2 - n2
    rhoL0, rhoV0 = anc[0].rhoL(T) , anc[0].rhoV(T)
    j = bin_mix.trace_VLE_isotherm_binary(T, np.array([rhoL0, 0]), np.array([rhoV0, 0]))
    df = pd.DataFrame(j) 
    df['Difference'] = (df["pL / Pa"] - p).abs()
    min_diff_index = df['Difference'].idxmin()
    closest_row = df.loc[min_diff_index]
    res1   = bin_mix.mix_VLE_Tp(T,p,np.array(closest_row["rhoL / mol/m^3"]),np.array(closest_row["rhoV / mol/m^3"]))
    rhoV_3 = np.append(res1.rhovecV,0.0)
    rhoL_3 = np.append(res1.rhovecL,0.0)

    # Provide binary solution as start for ternary trace
    ax = fig.add_subplot(1, 2, 1 + idx, projection='ternary')
    data = trace_VLE_isotherm_ternary(tern_mix, p , T, rhoL_3,rhoV_3)
    data = pd.DataFrame(data)
    # Plot the data using mpltern
    ax.plot(data["xL3"],data["xL1"],data["xL2"],label="saturated liquid curve")
    ax.plot(data["xV3"],data["xV1"],data["xV2"],label="saturated vapor  curve")
    ax.set_llabel("$x_{\mathrm{CO2}},y_{\mathrm{CO2}}$")
    ax.set_rlabel("$x_{\mathrm{N2}},y_{\mathrm{N2}}$")
    ax.set_tlabel("$x_{\mathrm{CH4}},y_{\mathrm{CH4}}$")
    ax.taxis.set_label_position('tick1')
    ax.laxis.set_label_position('tick1')
    ax.raxis.set_label_position('tick1')
    ax.set_title(f"T = {T} / K - p = {p/1E6} / MPa")
    ax.grid()
    
lines, labels = ax.get_legend_handles_labels()
fig.legend(lines, labels, loc='lower center', ncol=4)
plt.savefig("co2_n2_ch4.png")
