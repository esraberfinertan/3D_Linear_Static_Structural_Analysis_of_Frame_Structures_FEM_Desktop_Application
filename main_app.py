import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import sys

# Senin DBManager modülün varsa ekle, yoksa comment-out et
try:
    from database import DatabaseManager
except ImportError:
    DatabaseManager = None  # Demo için

# Senin solver'ın
solver_path = os.path.join(os.getcwd(), 'solver')
if solver_path not in sys.path:
    sys.path.insert(0, solver_path)
try:
    from run_solver import FrameSolver
except ImportError:
    FrameSolver = None

class FrameAnalysisApp:
    def __init__(self, user, project):
        self.user = user
        self.project = project
        self.db = DatabaseManager() if DatabaseManager else None

        self.root = tk.Tk()
        self.root.title(f"3D Frame Solver - {project['name']}")
        self.root.geometry("1000x800")

        self.nodes = None
        self.members = None

        self.analysis_figures = []
        self.current_result_index = 0

        self.setup_gui()
        self.setup_result_navigation()
        self.load_project_data()

    def setup_gui(self):
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)

        user_frame = ttk.Frame(control_frame)
        user_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(user_frame, text=f"User: {self.user['name']} | Project: {self.project['name']}", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="Load from CSV", command=self.load_from_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Project", command=self.save_project).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export to CSV", command=self.export_to_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Back to Projects", command=self.back_to_projects).pack(side=tk.LEFT, padx=5)

        model_frame = ttk.LabelFrame(self.root, text="Model Selection", padding="10")
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(model_frame, text="Predefined Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value="modelA")
        model_cb = ttk.Combobox(model_frame, textvariable=self.model_var, values=["modelA", "modelB"], state="readonly")
        model_cb.pack(side=tk.LEFT, padx=10)
        ttk.Button(model_frame, text="Load Model", command=self.load_predefined_model).pack(side=tk.LEFT, padx=10)
        ttk.Button(model_frame, text="Run Analysis", command=self.run_analysis).pack(side=tk.LEFT, padx=10)

        # Create the matplotlib canvas
        self.fig = Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initialize with a welcome message
        self.ax.text(0.5, 0.5, 0.5, "Welcome to 3D Frame Analysis\nLoad a model and run analysis to see results", 
                    ha='center', va='center', transform=self.ax.transAxes, fontsize=12)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self.ax.set_title("3D Frame Analysis")
        self.canvas.draw()

        self.status_var = tk.StringVar(value="Ready - Load a model and run analysis")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def setup_result_navigation(self):
        nav_frame = ttk.Frame(self.root)
        nav_frame.pack(fill=tk.X, pady=2)
        self.btn_prev = ttk.Button(nav_frame, text="← Previous", command=self.show_prev_result, state=tk.DISABLED)
        self.btn_prev.pack(side=tk.LEFT, padx=10)
        self.btn_next = ttk.Button(nav_frame, text="Next →", command=self.show_next_result, state=tk.DISABLED)
        self.btn_next.pack(side=tk.LEFT, padx=10)
        self.result_index_label = ttk.Label(nav_frame, text="No analysis results")
        self.result_index_label.pack(side=tk.LEFT, padx=20)
        
        # Add a label to show what type of output is currently displayed
        self.output_type_label = ttk.Label(nav_frame, text="", font=('Arial', 10, 'bold'))
        self.output_type_label.pack(side=tk.RIGHT, padx=20)

    def show_analysis_results(self, figures):
        if not figures:
            messagebox.showwarning("Uyarı", "Gösterilecek analiz sonucu yok.")
            return
        
        print(f"Showing {len(figures)} analysis results in GUI")
        self.analysis_figures = figures
        self.current_result_index = 0
        
        # Clear the current canvas and display the first figure
        self.update_analysis_canvas()
        self.update_nav_buttons()
        
        print("Analysis results displayed successfully")

    def update_analysis_canvas(self):
        if not self.analysis_figures:
            return
        
        print(f"Updating canvas with figure {self.current_result_index + 1}")
        fig = self.analysis_figures[self.current_result_index]
        
        # Destroy the old canvas widget and create a new one
        self.canvas.get_tk_widget().destroy()
        
        # Create new canvas with the figure
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Update the result label with plot name
        plot_name = self.plot_names[self.current_result_index] if hasattr(self, 'plot_names') and self.plot_names else f"Result {self.current_result_index + 1}"
        self.result_index_label.config(
            text=f"{plot_name} ({self.current_result_index + 1} / {len(self.analysis_figures)})"
        )
        
        print(f"Canvas updated with {plot_name}")

    def show_next_result(self):
        if self.analysis_figures and self.current_result_index < len(self.analysis_figures) - 1:
            self.current_result_index += 1
            self.update_analysis_canvas()
            self.update_nav_buttons()

    def show_prev_result(self):
        if self.analysis_figures and self.current_result_index > 0:
            self.current_result_index -= 1
            self.update_analysis_canvas()
            self.update_nav_buttons()

    def update_nav_buttons(self):
        total = len(self.analysis_figures)
        self.btn_prev.config(state=tk.NORMAL if self.current_result_index > 0 else tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL if self.current_result_index < total - 1 else tk.DISABLED)
        
        if total > 0:
            plot_name = self.plot_names[self.current_result_index] if hasattr(self, 'plot_names') and self.plot_names else f"Output {self.current_result_index + 1}"
            self.result_index_label.config(text=f"{plot_name} ({self.current_result_index + 1} / {total})")
            self.output_type_label.config(text=f"📊 {plot_name}")
        else:
            self.result_index_label.config(text="No analysis results")
            self.output_type_label.config(text="")

    def import_csv(self, path, dtype=float):
        try:
            return np.genfromtxt(path, delimiter=',', dtype=dtype)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {str(e)}")
            return None

    def load_predefined_model(self):
        model_name = self.model_var.get()
        base = f"models/{model_name}/"
        if not os.path.exists(base):
            messagebox.showerror("Error", f"Model directory '{model_name}' not found.")
            return
        vertices_file = base + "Vertices.csv"
        edges_file = base + "Edges.csv"
        if not os.path.exists(vertices_file) or not os.path.exists(edges_file):
            messagebox.showerror("Error", f"Required files not found in '{model_name}' directory.")
            return
        self.nodes = self.import_csv(vertices_file)
        self.members = self.import_csv(edges_file, int)
        if self.nodes is not None and self.members is not None:
            self.status_var.set(f"Loaded model: {model_name}")
            self.plot_structure()
        else:
            self.status_var.set("Failed to load model")

    def load_from_csv(self):
        vertices_file = filedialog.askopenfilename(title="Select Vertices CSV file", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not vertices_file:
            return
        edges_file = filedialog.askopenfilename(title="Select Edges CSV file", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not edges_file:
            return
        self.nodes = self.import_csv(vertices_file)
        self.members = self.import_csv(edges_file, int)
        if self.nodes is not None and self.members is not None:
            self.status_var.set("Loaded structure from CSV files")
            self.plot_structure()
        else:
            self.status_var.set("Failed to load CSV files")

    def plot_structure(self):
        if self.nodes is None or self.members is None:
            return
        self.ax.clear()
        for member in self.members:
            i, j = member
            i_idx, j_idx = int(i) - 1, int(j) - 1
            if i_idx < len(self.nodes) and j_idx < len(self.nodes):
                x = [self.nodes[i_idx][0], self.nodes[j_idx][0]]
                y = [self.nodes[i_idx][1], self.nodes[j_idx][1]]
                z = [self.nodes[i_idx][2], self.nodes[j_idx][2]]
                self.ax.plot(x, y, z, 'b-', linewidth=2)
        for idx, node in enumerate(self.nodes):
            self.ax.scatter(node[0], node[1], node[2], color='red', s=50)
            self.ax.text(node[0], node[1], node[2], str(idx + 1), color='red', fontsize=8)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title(f"3D Structure - {self.project['name']}")
        self.canvas.draw()
        self.analysis_figures = []
        self.current_result_index = 0
        self.update_nav_buttons()

    def save_project(self):
        if self.db is None:
            messagebox.showwarning("Warning", "DatabaseManager modülü bulunamadı!")
            return
        if self.nodes is None or self.members is None:
            messagebox.showwarning("Warning", "No structure data to save.")
            return
        try:
            nodes_list = [tuple(node) for node in self.nodes]
            members_list = [tuple(member) for member in self.members]
            self.db.save_project_data(self.project['id'], nodes_list, members_list)
            self.status_var.set("Project saved successfully")
            messagebox.showinfo("Success", "Project saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save project: {str(e)}")

    def load_project_data(self):
        if self.db is None:
            self.status_var.set("DB modülü yok, demo çalışıyor.")
            return
        try:
            nodes, members = self.db.load_project_data(self.project['id'])
            if nodes and members:
                self.nodes = np.array(nodes)
                self.members = np.array(members)
                self.plot_structure()
                self.status_var.set("Project data loaded")
            else:
                self.status_var.set("No saved data found for this project")
        except Exception as e:
            self.status_var.set(f"Error loading project data: {str(e)}")

    def export_to_csv(self):
        if self.nodes is None or self.members is None:
            messagebox.showwarning("Warning", "No structure data to export.")
            return
        export_dir = filedialog.askdirectory(title="Select directory to export CSV files")
        if not export_dir:
            return
        try:
            vertices_file = os.path.join(export_dir, "Vertices.csv")
            np.savetxt(vertices_file, self.nodes, delimiter=',', fmt='%.6f')
            edges_file = os.path.join(export_dir, "Edges.csv")
            np.savetxt(edges_file, self.members, delimiter=',', fmt='%d')
            messagebox.showinfo("Success", f"Files exported to:\n{vertices_file}\n{edges_file}")
            self.status_var.set("Structure exported to CSV")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export files: {str(e)}")

    def run_analysis(self):
        if self.nodes is None or self.members is None:
            messagebox.showwarning("Warning", "Please load a structure first.")
            return
        try:
            analysis_window = tk.Toplevel(self.root)
            analysis_window.title("Analysis Options")
            analysis_window.geometry("400x300")
            analysis_window.resizable(False, False)
            analysis_window.transient(self.root)
            analysis_window.grab_set()
            ttk.Label(analysis_window, text="Select Analysis Outputs:", font=('Arial', 12, 'bold')).pack(pady=10)

            self.show_structure = tk.BooleanVar(value=True)
            self.show_deflection = tk.BooleanVar(value=True)
            self.show_axial_forces = tk.BooleanVar(value=True)
            self.show_bmd = tk.BooleanVar(value=True)
            self.show_sfd = tk.BooleanVar(value=True)
            self.show_text_output = tk.BooleanVar(value=True)

            ttk.Checkbutton(analysis_window, text="Original Structure", variable=self.show_structure).pack(anchor=tk.W, padx=20, pady=2)
            ttk.Checkbutton(analysis_window, text="Deflected Shape", variable=self.show_deflection).pack(anchor=tk.W, padx=20, pady=2)
            ttk.Checkbutton(analysis_window, text="Axial Forces", variable=self.show_axial_forces).pack(anchor=tk.W, padx=20, pady=2)
            ttk.Checkbutton(analysis_window, text="Bending Moment Diagram (BMD)", variable=self.show_bmd).pack(anchor=tk.W, padx=20, pady=2)
            ttk.Checkbutton(analysis_window, text="Shear Force Diagram (SFD)", variable=self.show_sfd).pack(anchor=tk.W, padx=20, pady=2)
            ttk.Checkbutton(analysis_window, text="Text Output (Console)", variable=self.show_text_output).pack(anchor=tk.W, padx=20, pady=2)
            button_frame = ttk.Frame(analysis_window)
            button_frame.pack(pady=20)
            ttk.Button(button_frame, text="Run Analysis", command=lambda: self.execute_analysis(analysis_window)).pack(side=tk.LEFT, padx=10)
            ttk.Button(button_frame, text="Cancel", command=analysis_window.destroy).pack(side=tk.LEFT, padx=10)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to setup analysis: {str(e)}")

    def execute_analysis(self, dialog_window):
        dialog_window.destroy()
        try:
            if FrameSolver is None:
                messagebox.showwarning("Solver Eksik", "Solver bulunamadı!")
                return
            
            print("Creating FrameSolver...")
            solver = FrameSolver(model=self.model_var.get())
            
            self.status_var.set("Running structural analysis...")
            print("Running analyze()...")
            results = solver.analyze()
            print(f"Analysis results: {results}")
            
            figures = []
            plot_names = []
            
            print("Creating plots...")
            
            # 0. Original Structure
            if self.show_structure.get():
                print("Creating structure plot...")
                try:
                    fig = solver.plot_structure()
                    if fig is not None:
                        figures.append(fig)
                        plot_names.append("Original Structure")
                        print("Structure plot added")
                    else:
                        print("Structure plot returned None")
                except Exception as e:
                    print(f"Error creating structure plot: {e}")
                
            # 1. Deflected Shape
            if self.show_deflection.get():
                print("Creating deflection plot...")
                try:
                    fig = solver.plot_deflection()
                    if fig is not None:
                        figures.append(fig)
                        plot_names.append("Deflected Shape")
                        print("Deflection plot added")
                    else:
                        print("Deflection plot returned None")
                except Exception as e:
                    print(f"Error creating deflection plot: {e}")
                
            # 2. Axial Forces
            if self.show_axial_forces.get():
                print("Creating axial forces plot...")
                try:
                    fig = solver.plot_axial_forces()
                    if fig is not None:
                        figures.append(fig)
                        plot_names.append("Axial Forces")
                        print("Axial forces plot added")
                    else:
                        print("Axial forces plot returned None")
                except Exception as e:
                    print(f"Error creating axial forces plot: {e}")
                
            # 3. Bending Moment Diagram
            if self.show_bmd.get():
                print("Creating BMD plot...")
                try:
                    fig = solver.plot_bmd()
                    if fig is not None:
                        figures.append(fig)
                        plot_names.append("Bending Moment Diagram")
                        print("BMD plot added")
                    else:
                        print("BMD plot returned None")
                except Exception as e:
                    print(f"Error creating BMD plot: {e}")
                
            # 4. Shear Force Diagram
            if self.show_sfd.get():
                print("Creating SFD plot...")
                try:
                    fig = solver.plot_sfd()
                    if fig is not None:
                        figures.append(fig)
                        plot_names.append("Shear Force Diagram")
                        print("SFD plot added")
                    else:
                        print("SFD plot returned None")
                except Exception as e:
                    print(f"Error creating SFD plot: {e}")
                
            if self.show_text_output.get():
                print("Printing results...")
                solver.print_results()
                
            print(f"Total figures created: {len(figures)}")
            print(f"Plot names: {plot_names}")
            
            if figures:
                # Store plot names for display
                self.plot_names = plot_names
                self.show_analysis_results(figures)
                
                # Show success message with details
                result_text = f"Structural analysis completed successfully!\n\n"
                result_text += f"Nodes: {len(self.nodes) if self.nodes is not None else 0}\n"
                result_text += f"Members: {len(self.members) if self.members is not None else 0}\n"
                if results:
                    result_text += f"Max Displacement: {results.get('max_displacement', 'N/A'):.6f} m\n"
                    result_text += f"Max Stress: {results.get('max_stress', 'N/A'):.2f} Pa\n"
                result_text += f"Figures created: {len(figures)}\n\n"
                result_text += "Available outputs:\n"
                for i, name in enumerate(plot_names):
                    result_text += f"{i+1}. {name}\n"
                
                messagebox.showinfo("Analysis Complete", result_text)
                self.status_var.set(f"Analysis completed - {len(figures)} outputs available")
            else:
                messagebox.showwarning("No Figures", "No figures were created. Check the analysis options.")
                self.status_var.set("Analysis completed but no figures created")
                
        except Exception as e:
            import traceback
            error_msg = f"Analysis failed: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(error_msg)
            messagebox.showerror("Analysis Error", error_msg)
            self.status_var.set("Analysis failed")

    def back_to_projects(self):
        self.root.destroy()
        try:
            from auth_gui import AuthenticationGUI
            auth_app = AuthenticationGUI()
            auth_app.current_user = self.user
            auth_app.show_projects_screen()
            auth_app.run()
        except ImportError:
            pass

    def run(self):
        self.root.mainloop()


# --- TEST (main) ---
if __name__ == "__main__":
    test_user = {'id': 1, 'name': 'Test User', 'email': 'test@example.com'}
    test_project = {'id': 1, 'name': 'Test Project', 'description': 'Test project'}
    app = FrameAnalysisApp(test_user, test_project)
    app.run()
