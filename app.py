#!/usr/bin/env python3


import sys
import os
import tkinter as tk
from tkinter import messagebox

def check_dependencies():
    """Check if all required dependencies are installed"""
    missing_deps = []
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")
    
    try:
        import flask
    except ImportError:
        missing_deps.append("flask")
    
    try:
        import google_auth_oauthlib
    except ImportError:
        missing_deps.append("google-auth-oauthlib")
    
    if missing_deps:
        print("Missing dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nPlease install missing dependencies:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main entry point of the application"""
    print("Frame Analysis Application")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    try:
        # Import and start the authentication GUI
        from auth_gui import AuthenticationGUI
        
        print("Starting authentication system...")
        app = AuthenticationGUI()
        app.run()
        
    except ImportError as e:
        messagebox.showerror("Import Error", 
                           f"Failed to import required modules:\n{str(e)}\n\n"
                           "Please ensure all dependencies are installed:\n"
                           "pip install -r requirements.txt")
        sys.exit(1)
    
    except Exception as e:
        messagebox.showerror("Application Error", 
                           f"An unexpected error occurred:\n{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 