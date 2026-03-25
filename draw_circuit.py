"""
Publication circuit diagram using PennyLane draw_mpl with per-wire colors.
Only draws 1 layer for clarity.
"""
import pennylane as qml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

BLUE = '#1565C0'
GREEN = '#2E7D32'
ORANGE = '#E65100'

# Use named wires for readable labels
wire_names = [
    "δ (PSD)", "θ (PSD)", "α (PSD)", "β (PSD)", "γ (PSD)",
    "DASM₁", "DASM₂", "DASM₃", "DASM₄",
    "PLV₁", "PLV₂", "PLV₃",
]

n_qubits = 12

x = np.random.rand(n_qubits) * 0.5

# Build a tape with no measurements (no measurement symbols drawn)
ops = []
# Feature encoding
for i, w in enumerate(wire_names):
    ops.append(qml.RY(x[i] * np.pi, wires=w))

# Cross-frequency coupling
ops.append(qml.CNOT(wires=["θ (PSD)", "γ (PSD)"]))
ops.append(qml.RZ(x[1] * x[4], wires="γ (PSD)"))
ops.append(qml.CNOT(wires=["α (PSD)", "β (PSD)"]))
ops.append(qml.RZ(x[2] * x[3], wires="β (PSD)"))
ops.append(qml.CNOT(wires=["δ (PSD)", "θ (PSD)"]))

# Hemispheric asymmetry
ops.append(qml.CNOT(wires=["DASM₁", "α (PSD)"]))
ops.append(qml.CNOT(wires=["DASM₂", "β (PSD)"]))
ops.append(qml.CNOT(wires=["DASM₁", "DASM₂"]))
ops.append(qml.CNOT(wires=["DASM₂", "DASM₃"]))
ops.append(qml.CNOT(wires=["DASM₃", "DASM₄"]))

# Connectivity modulation
ops.append(qml.CRZ(x[9], wires=["PLV₁", "DASM₁"]))
ops.append(qml.CRZ(x[10], wires=["PLV₂", "DASM₂"]))
ops.append(qml.CRZ(x[11], wires=["PLV₃", "DASM₃"]))

# Inter-layer re-encoding
for i, w in enumerate(wire_names):
    ops.append(qml.RX(x[i] * np.pi * 0.5, wires=w))

tape = qml.tape.QuantumTape(ops, [])

# Per-wire coloring
wire_options = {
    "linewidth": 1.0, "color": "black",
    "δ (PSD)": {"color": BLUE, "linewidth": 1.2},
    "θ (PSD)": {"color": BLUE, "linewidth": 1.2},
    "α (PSD)": {"color": BLUE, "linewidth": 1.2},
    "β (PSD)": {"color": BLUE, "linewidth": 1.2},
    "γ (PSD)": {"color": BLUE, "linewidth": 1.2},
    "DASM₁": {"color": GREEN, "linewidth": 1.2},
    "DASM₂": {"color": GREEN, "linewidth": 1.2},
    "DASM₃": {"color": GREEN, "linewidth": 1.2},
    "DASM₄": {"color": GREEN, "linewidth": 1.2},
    "PLV₁": {"color": ORANGE, "linewidth": 1.2},
    "PLV₂": {"color": ORANGE, "linewidth": 1.2},
    "PLV₃": {"color": ORANGE, "linewidth": 1.2},
}

qml.drawer.use_style("black_white")

fig, ax = qml.drawer.tape_mpl(
    tape,
    wire_order=wire_names,
    show_all_wires=True,
    decimals=None,
    wire_options=wire_options,
    label_options={"fontsize": 9},
)

# Add legend for wire groups
legend_elements = [
    mpatches.Patch(facecolor=BLUE, alpha=0.25, edgecolor=BLUE,
                   label='Frequency bands (δ, θ, α, β, γ)'),
    mpatches.Patch(facecolor=GREEN, alpha=0.25, edgecolor=GREEN,
                   label='Hemispheric asymmetry (DASM)'),
    mpatches.Patch(facecolor=ORANGE, alpha=0.25, edgecolor=ORANGE,
                   label='Connectivity (PLV)'),
]
ax.legend(handles=legend_elements, loc='upper center', fontsize=16,
          frameon=True, framealpha=0.95, edgecolor='#ccc', ncol=3,
          bbox_to_anchor=(0.5, -0.03))

fig.set_size_inches(14, 6)

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/circuit_informed.pdf", format="pdf", dpi=300, bbox_inches='tight')
fig.savefig("figures/circuit_informed.png", format="png", dpi=300, bbox_inches='tight')
print("Saved to figures/circuit_informed.pdf and .png")
plt.close()