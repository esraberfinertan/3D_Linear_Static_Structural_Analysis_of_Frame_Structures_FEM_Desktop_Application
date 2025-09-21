#!/usr/bin/env python3
"""
Test script to verify FrameSolver functionality
"""

import sys
import os

# Add solver path to sys.path
solver_path = os.path.join(os.getcwd(), 'solver')
if solver_path not in sys.path:
    sys.path.insert(0, solver_path)

try:
    from run_solver import FrameSolver
    print("✅ FrameSolver imported successfully")
except ImportError as e:
    print(f"❌ Failed to import FrameSolver: {e}")
    sys.exit(1)

def test_solver():
    print("\n🧪 Testing FrameSolver...")
    
    try:
        # Create solver instance
        solver = FrameSolver(model="modelA")
        print("✅ FrameSolver instance created")
        
        # Run analysis
        results = solver.analyze()
        print(f"✅ Analysis completed: {results}")
        
        # Test all plotting methods
        figures = []
        
        print("📊 Testing plot_structure...")
        fig1 = solver.plot_structure()
        figures.append(fig1)
        print("✅ Structure plot created")
        
        print("📊 Testing plot_deflection...")
        fig2 = solver.plot_deflection()
        figures.append(fig2)
        print("✅ Deflection plot created")
        
        print("📊 Testing plot_axial_forces...")
        fig3 = solver.plot_axial_forces()
        figures.append(fig3)
        print("✅ Axial forces plot created")
        
        print("📊 Testing plot_bmd...")
        fig4 = solver.plot_bmd()
        figures.append(fig4)
        print("✅ BMD plot created")
        
        print("📊 Testing plot_sfd...")
        fig5 = solver.plot_sfd()
        figures.append(fig5)
        print("✅ SFD plot created")
        
        print(f"✅ All plots created successfully! Total: {len(figures)}")
        
        # Test print_results
        print("\n📝 Testing print_results...")
        solver.print_results()
        print("✅ Results printed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_solver()
    if success:
        print("\n🎉 All tests passed! FrameSolver is working correctly.")
    else:
        print("\n💥 Tests failed! There are issues with FrameSolver.")
        sys.exit(1) 