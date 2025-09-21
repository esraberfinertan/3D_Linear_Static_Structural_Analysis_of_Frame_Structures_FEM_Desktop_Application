from fea import analyze
import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
figures = []
current_index = 0


# === CSV yükleyici ===
def import_csv(path, dtype=float):
    return np.genfromtxt(path, delimiter=',', dtype=dtype)

# === Model yükleyici ===
def load_model(model_name):
    base = f"models/{model_name}/"
    global nodes, members, loads, supports
    nodes = import_csv(base + "Vertices.csv")
    members = import_csv(base + "Edges.csv", int)
    loads = import_csv(base + "Loads.csv")
    supports = import_csv(base + "Supports.csv", int)
    print(f"{model_name} modeli yüklendi.")


# === Yapıyı çizen fonksiyon ===
def plotStructure(ax):
    ax.clear()
    for m in members:
        i, j = m
        x = [nodes[i-1][0], nodes[j-1][0]]
        y = [nodes[i-1][1], nodes[j-1][1]]
        z = [nodes[i-1][2], nodes[j-1][2]]
        ax.plot(x, y, z, 'b')
    for idx, node in enumerate(nodes):
        ax.text(node[0], node[1], node[2], str(idx+1), color='red')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Yapı Görselleştirmesi")
    canvas.draw()

# === Tkinter GUI ===
root = tk.Tk()
root.title("3D Frame Solver")
# === Sonuçları gösterecek kutu ===
result_box = tk.Text(root, height=20, width=70)
result_box.grid(row=2, column=0, columnspan=3, padx=10, pady=10)


# === Model seçimi ===
tk.Label(root, text="Model:").grid(row=0, column=0)
#model_cb = ttk.Combobox(root, values=["modelA", "modelB"])
model_cb = ttk.Combobox(root, values=["modelA", "modelB", "modelC"])

model_cb.set("modelA")
model_cb.grid(row=0, column=1)

# === Matplotlib Sahnesi ===
fig = Figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=1, column=0, columnspan=3, pady=10)
nav_frame = tk.Frame(root)
nav_frame.grid(row=2, column=0, columnspan=3, pady=5)

tk.Button(nav_frame, text="← Önceki", command=previous_plot).pack(side=tk.LEFT, padx=10)
tk.Button(nav_frame, text="Sonraki →", command=next_plot).pack(side=tk.LEFT, padx=10)

def update_canvas():
    global canvas
    canvas.get_tk_widget().destroy()  # Eski canvas'ı sil
    fig = figures[current_index]
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=1, column=0, columnspan=3, pady=10)
    canvas.draw()

figures = []
current_index = 0

# === Run butonu ===
def run_analysis():
    selected_model = model_cb.get()
    load_model(selected_model)

    global figures, current_index
    figures = []

    # 1. Original Structure
    fig1 = Figure(figsize=(6, 5))
    ax1 = fig1.add_subplot(111, projection='3d')
    plotStructure(ax1)
    ax1.set_title("3D Structure")
    figures.append(fig1)

    # 2. Deflected Shape
    fig2 = Figure(figsize=(6, 5))
    ax2 = fig2.add_subplot(111, projection='3d')
    # → Buraya kendi deformasyon çizim fonksiyonunu yaz
    plot_deflected_shape(ax2)
    ax2.set_title("Deflected Shape")
    figures.append(fig2)

    # 3. Axial Forces
    fig3 = Figure(figsize=(6, 5))
    ax3 = fig3.add_subplot(111, projection='3d')
    # → Buraya axial force çizim fonksiyonunu yaz
    plot_axial_forces(ax3)
    ax3.set_title("Axial Forces")
    figures.append(fig3)

    # 4. Bending Moments
    fig4 = Figure(figsize=(6, 5))
    ax4 = fig4.add_subplot(111, projection='3d')
    # → Buraya moment çizim fonksiyonunu yaz
    plot_moments(ax4)
    ax4.set_title("Bending Moment")
    figures.append(fig4)

    # 5. Shear Forces
    fig5 = Figure(figsize=(6, 5))
    ax5 = fig5.add_subplot(111, projection='3d')
    # → Buraya shear force çizim fonksiyonunu yaz
    plot_shear_forces(ax5)
    ax5.set_title("Shear Forces")
    figures.append(fig5)

    current_index = 0
    update_canvas()
    print("✅ Analiz tamamlandı.")
    print("📊 Düğümlerin yer değiştirmeleri:")
    print(displacements)



#tk.Button(root, text="ModelC Oluştur", command=generate_modelC).grid(row=1, column=2)
tk.Button(root, text="Run", command=run_analysis).grid(row=0, column=2)


def generate_original_structure():
    fig = Figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Original Structure")
    for m in members:
        i, j = m
        xi, xj = nodes[i-1], nodes[j-1]
        ax.plot([xi[0], xj[0]], [xi[1], xj[1]], [xi[2], xj[2]], 'b')
    for idx, n in enumerate(nodes):
        ax.scatter(n[0], n[1], n[2], color='red')
        ax.text(n[0], n[1], n[2], str(idx+1), color='black', fontsize=6)
    return fig

def generate_deflected_shape():
    fig = Figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Deflected Shape (Scale=100)")
    scale = 100
    for m in members:
        i, j = m
        xi = nodes[i-1] + scale * displacements[i-1]
        xj = nodes[j-1] + scale * displacements[j-1]
        ax.plot([xi[0], xj[0]], [xi[1], xj[1]], [xi[2], xj[2]], 'r')
    return fig

def generate_axial_forces():
    fig = Figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Axial Forces (Red=Tension, Blue=Compression)")
    for idx, m in enumerate(members):
        i, j = m
        xi, xj = nodes[i-1], nodes[j-1]
        color = 'r' if axial_forces[idx] >= 0 else 'b'
        ax.plot([xi[0], xj[0]], [xi[1], xj[1]], [xi[2], xj[2]], color)
        cx = (xi[0] + xj[0]) / 2
        cy = (xi[1] + xj[1]) / 2
        cz = (xi[2] + xj[2]) / 2
        ax.text(cx, cy, cz, f"{int(axial_forces[idx])}N", fontsize=6)
    return fig

def generate_bending_moment():
    fig = Figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Bending Moment Diagram")
    for idx, m in enumerate(members):
        i, j = m
        xi, xj = nodes[i-1], nodes[j-1]
        lw = max(1, abs(bending_moments[idx]) / 50000)
        ax.plot([xi[0], xj[0]], [xi[1], xj[1]], [xi[2], xj[2]], 'b', linewidth=lw)
    return fig

def generate_shear_force():
    fig = Figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Shear Force Diagram")
    for idx, m in enumerate(members):
        i, j = m
        xi, xj = nodes[i-1], nodes[j-1]
        lw = max(1, abs(shear_forces[idx]) / 500)
        ax.plot([xi[0], xj[0]], [xi[1], xj[1]], [xi[2], xj[2]], 'g', linewidth=lw)
    return fig

def update_canvas():
    global canvas, current_index
    fig = figures[current_index]
    canvas.figure = fig
    canvas.draw()

def next_plot():
    global current_index
    if current_index < len(figures) - 1:
        current_index += 1
        update_canvas()

def previous_plot():
    global current_index
    if current_index > 0:
        current_index -= 1
        update_canvas()


root.mainloop()
