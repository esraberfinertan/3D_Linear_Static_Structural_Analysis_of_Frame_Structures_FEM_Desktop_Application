import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading
from database import DatabaseManager
import re

# Try importing Google auth with fallback
try:
    from google_auth import authenticate_with_google
    GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    GOOGLE_AUTH_AVAILABLE = False
    def authenticate_with_google():
        # Demo implementation when Google auth is not available
        return {
            'id': 'demo_google_id_123',
            'name': 'Demo User',
            'email': 'demo@example.com'
        }

class AuthenticationGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Frame Analysis - Authentication")
        self.root.geometry("400x500")
        self.root.resizable(False, False)
        
        # Initialize database
        self.db = DatabaseManager()
        
        # Current user info
        self.current_user = None
        
        # Style configuration
        self.setup_styles()
        
        # Start with login screen
        self.show_login_screen()
    
    def setup_styles(self):
        """Setup GUI styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Heading.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Large.TButton', font=('Arial', 10), padding=10)
    
    def clear_window(self):
        """Clear all widgets from the window"""
        for widget in self.root.winfo_children():
            widget.destroy()
    
    def show_login_screen(self):
        """Display the login screen"""
        self.clear_window()
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Frame Analysis App", style='Title.TLabel')
        title_label.pack(pady=(0, 30))
        
        # Login form frame
        login_frame = ttk.LabelFrame(main_frame, text="Sign In", padding="20")
        login_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Email field
        ttk.Label(login_frame, text="Email:").pack(anchor=tk.W)
        self.email_entry = ttk.Entry(login_frame, width=30, font=('Arial', 10))
        self.email_entry.pack(fill=tk.X, pady=(5, 15))
        
        # Password field
        ttk.Label(login_frame, text="Password:").pack(anchor=tk.W)
        self.password_entry = ttk.Entry(login_frame, width=30, show='*', font=('Arial', 10))
        self.password_entry.pack(fill=tk.X, pady=(5, 15))
        
        # Login button
        login_btn = ttk.Button(login_frame, text="Sign In", command=self.handle_login, style='Large.TButton')
        login_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Google login button
        google_text = "Sign In with Google" if GOOGLE_AUTH_AVAILABLE else "Sign In with Google (Demo)"
        google_btn = ttk.Button(login_frame, text=google_text, 
                               command=self.handle_google_login, style='Large.TButton')
        google_btn.pack(fill=tk.X)
        
        if not GOOGLE_AUTH_AVAILABLE:
            ttk.Label(login_frame, text="Note: Google auth unavailable, using demo mode", 
                     font=('Arial', 8), foreground='gray').pack(pady=(5, 0))
        
        # Separator
        separator = ttk.Separator(main_frame, orient=tk.HORIZONTAL)
        separator.pack(fill=tk.X, pady=20)
        
        # Register section
        register_frame = ttk.Frame(main_frame)
        register_frame.pack(fill=tk.X)
        
        ttk.Label(register_frame, text="Don't have an account?").pack()
        register_btn = ttk.Button(register_frame, text="Create Account", 
                                 command=self.show_register_screen)
        register_btn.pack(pady=(10, 0))
        
        # Bind Enter key to login
        self.root.bind('<Return>', lambda e: self.handle_login())
        
        # Focus on email entry
        self.email_entry.focus()
    
    def show_register_screen(self):
        """Display the registration screen"""
        self.clear_window()
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Back button
        back_btn = ttk.Button(main_frame, text="← Back to Sign In", command=self.show_login_screen)
        back_btn.pack(anchor=tk.W, pady=(0, 20))
        
        # Title
        title_label = ttk.Label(main_frame, text="Create Account", style='Title.TLabel')
        title_label.pack(pady=(0, 30))
        
        # Registration form
        register_frame = ttk.LabelFrame(main_frame, text="New Account", padding="20")
        register_frame.pack(fill=tk.X)
        
        # Name field
        ttk.Label(register_frame, text="Full Name:").pack(anchor=tk.W)
        self.reg_name_entry = ttk.Entry(register_frame, width=30, font=('Arial', 10))
        self.reg_name_entry.pack(fill=tk.X, pady=(5, 15))
        
        # Email field
        ttk.Label(register_frame, text="Email:").pack(anchor=tk.W)
        self.reg_email_entry = ttk.Entry(register_frame, width=30, font=('Arial', 10))
        self.reg_email_entry.pack(fill=tk.X, pady=(5, 15))
        
        # Password field
        ttk.Label(register_frame, text="Password:").pack(anchor=tk.W)
        self.reg_password_entry = ttk.Entry(register_frame, width=30, show='*', font=('Arial', 10))
        self.reg_password_entry.pack(fill=tk.X, pady=(5, 15))
        
        # Confirm Password field
        ttk.Label(register_frame, text="Confirm Password:").pack(anchor=tk.W)
        self.reg_confirm_entry = ttk.Entry(register_frame, width=30, show='*', font=('Arial', 10))
        self.reg_confirm_entry.pack(fill=tk.X, pady=(5, 15))
        
        # Register button
        register_btn = ttk.Button(register_frame, text="Create Account", 
                                 command=self.handle_register, style='Large.TButton')
        register_btn.pack(fill=tk.X)
        
        # Bind Enter key to register
        self.root.bind('<Return>', lambda e: self.handle_register())
        
        # Focus on name entry
        self.reg_name_entry.focus()
    
    def validate_email(self, email):
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def handle_login(self):
        """Handle email/password login"""
        email = self.email_entry.get().strip()
        password = self.password_entry.get()
        
        if not email or not password:
            messagebox.showerror("Error", "Please enter both email and password.")
            return
        
        if not self.validate_email(email):
            messagebox.showerror("Error", "Please enter a valid email address.")
            return
        
        # Authenticate user
        user = self.db.authenticate_user(email, password)
        if user:
            self.current_user = user
            self.show_projects_screen()
        else:
            messagebox.showerror("Error", "Invalid email or password.")
    
    def handle_register(self):
        """Handle user registration"""
        name = self.reg_name_entry.get().strip()
        email = self.reg_email_entry.get().strip()
        password = self.reg_password_entry.get()
        confirm_password = self.reg_confirm_entry.get()
        
        # Validation
        if not all([name, email, password, confirm_password]):
            messagebox.showerror("Error", "Please fill in all fields.")
            return
        
        if not self.validate_email(email):
            messagebox.showerror("Error", "Please enter a valid email address.")
            return
        
        if len(password) < 6:
            messagebox.showerror("Error", "Password must be at least 6 characters long.")
            return
        
        if password != confirm_password:
            messagebox.showerror("Error", "Passwords do not match.")
            return
        
        # Check if user already exists
        existing_user = self.db.get_user_by_email(email)
        if existing_user:
            messagebox.showerror("Error", "An account with this email already exists.")
            return
        
        # Create user
        user_id = self.db.create_user(name, email, password)
        if user_id:
            messagebox.showinfo("Success", "Account created successfully! Please sign in.")
            self.show_login_screen()
        else:
            messagebox.showerror("Error", "Failed to create account. Please try again.")
    
    def handle_google_login(self):
        """Handle Google OAuth login"""
        def auth_thread():
            try:
                user_info = authenticate_with_google()
                if user_info:
                    # Check if user exists
                    existing_user = self.db.get_user_by_google_id(user_info['id'])
                    if not existing_user:
                        # Check if email exists with different auth method
                        email_user = self.db.get_user_by_email(user_info['email'])
                        if email_user:
                            self.root.after(0, lambda: messagebox.showerror(
                                "Error", 
                                "An account with this email already exists. Please sign in with email/password."
                            ))
                            return
                        
                        # Create new user
                        user_id = self.db.create_user(
                            user_info['name'], 
                            user_info['email'], 
                            google_id=user_info['id']
                        )
                        if user_id:
                            self.current_user = {
                                'id': user_id,
                                'name': user_info['name'],
                                'email': user_info['email']
                            }
                        else:
                            self.root.after(0, lambda: messagebox.showerror(
                                "Error", "Failed to create account."
                            ))
                            return
                    else:
                        self.current_user = existing_user
                    
                    # Show projects screen
                    self.root.after(0, self.show_projects_screen)
                else:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", "Google authentication failed."
                    ))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Authentication error: {str(e)}"
                ))
        
        # Show loading message
        auth_note = "Demo mode" if not GOOGLE_AUTH_AVAILABLE else "Opening browser for Google authentication"
        messagebox.showinfo("Google Sign In", f"{auth_note}...")
        
        # Start authentication in separate thread
        thread = threading.Thread(target=auth_thread)
        thread.daemon = True
        thread.start()
    
    def show_projects_screen(self):
        """Display the projects screen"""
        self.clear_window()
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header frame
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Welcome message
        welcome_label = ttk.Label(header_frame, 
                                 text=f"Welcome, {self.current_user['name']}!", 
                                 style='Title.TLabel')
        welcome_label.pack(side=tk.LEFT)
        
        # Logout button
        logout_btn = ttk.Button(header_frame, text="Logout", command=self.logout)
        logout_btn.pack(side=tk.RIGHT)
        
        # Projects section
        projects_frame = ttk.LabelFrame(main_frame, text="Your Projects", padding="15")
        projects_frame.pack(fill=tk.BOTH, expand=True)
        
        # Projects listbox with scrollbar
        list_frame = ttk.Frame(projects_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        self.projects_listbox = tk.Listbox(list_frame, font=('Arial', 10))
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.projects_listbox.yview)
        self.projects_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.projects_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Buttons frame
        buttons_frame = ttk.Frame(projects_frame)
        buttons_frame.pack(fill=tk.X)
        
        ttk.Button(buttons_frame, text="New Project", 
                  command=self.create_new_project).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(buttons_frame, text="Open Project", 
                  command=self.open_project).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(buttons_frame, text="Delete Project", 
                  command=self.delete_project).pack(side=tk.LEFT)
        
        # Load and display projects
        self.refresh_projects_list()
        
        # Bind double-click to open project
        self.projects_listbox.bind('<Double-1>', lambda e: self.open_project())
    
    def refresh_projects_list(self):
        """Refresh the projects list"""
        self.projects_listbox.delete(0, tk.END)
        projects = self.db.get_user_projects(self.current_user['id'])
        
        if not projects:
            self.projects_listbox.insert(0, "No projects yet. Create a new project to get started!")
        else:
            for project in projects:
                self.projects_listbox.insert(tk.END, f"{project['name']} - {project['description']}")
        
        self.projects = projects
    
    def create_new_project(self):
        """Create a new project"""
        name = simpledialog.askstring("New Project", "Enter project name:")
        if not name:
            return
        
        description = simpledialog.askstring("New Project", "Enter project description (optional):") or ""
        
        try:
            project_id = self.db.create_project(self.current_user['id'], name, description)
            if project_id:
                messagebox.showinfo("Success", f"Project '{name}' created successfully!")
                self.refresh_projects_list()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create project: {str(e)}")
    
    def open_project(self):
        """Open selected project"""
        selection = self.projects_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a project to open.")
            return
        
        if not hasattr(self, 'projects') or not self.projects:
            messagebox.showwarning("Warning", "No projects available.")
            return
        
        selected_idx = selection[0]
        if selected_idx >= len(self.projects):
            return
        
        project = self.projects[selected_idx]
        
        # Close auth window and open main app
        self.root.destroy()
        
        # Import and run main application
        try:
            from main_app import FrameAnalysisApp
            main_app = FrameAnalysisApp(self.current_user, project)
            main_app.run()
        except ImportError:
            messagebox.showerror("Error", "Main application module not found.")
    
    def delete_project(self):
        """Delete selected project"""
        selection = self.projects_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a project to delete.")
            return
        
        if not hasattr(self, 'projects') or not self.projects:
            return
        
        selected_idx = selection[0]
        if selected_idx >= len(self.projects):
            return
        
        project = self.projects[selected_idx]
        
        # Confirm deletion
        if messagebox.askyesno("Confirm Delete", 
                              f"Are you sure you want to delete project '{project['name']}'?\n"
                              "This action cannot be undone."):
            try:
                if self.db.delete_project(project['id'], self.current_user['id']):
                    messagebox.showinfo("Success", "Project deleted successfully!")
                    self.refresh_projects_list()
                else:
                    messagebox.showerror("Error", "Failed to delete project.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete project: {str(e)}")
    
    def logout(self):
        """Logout user"""
        self.current_user = None
        self.show_login_screen()
    
    def run(self):
        """Run the authentication GUI"""
        self.root.mainloop()

if __name__ == "__main__":
    app = AuthenticationGUI()
    app.run() 