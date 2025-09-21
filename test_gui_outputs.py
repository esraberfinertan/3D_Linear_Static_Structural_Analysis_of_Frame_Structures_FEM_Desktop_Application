#!/usr/bin/env python3
"""
Test script to verify that all 5 outputs from FrameSolver are working correctly
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

def test_all_outputs():
    print("\n🧪 Testing all 5 outputs from FrameSolver...")
    
    try:
        # Create solver instance
        solver = FrameSolver(model="modelA")
        print("✅ FrameSolver instance created")
        
        # Run analysis
        results = solver.analyze()
        print(f"✅ Analysis completed: {results}")
        
        # Test all 5 plotting methods
        outputs = [
            ("Original Structure", solver.plot_structure),
            ("Deflected Shape", solver.plot_deflection),
            ("Axial Forces", solver.plot_axial_forces),
            ("Bending Moment Diagram", solver.plot_bmd),
            ("Shear Force Diagram", solver.plot_sfd)
        ]
        
        figures = []
        for i, (name, plot_func) in enumerate(outputs):
            try:
                print(f"\n📊 Testing {name}...")
                fig = plot_func()
                if fig is not None:
                    figures.append(fig)
                    print(f"✅ {name} - Figure created successfully")
                    print(f"   Figure type: {type(fig)}")
                    print(f"   Figure size: {fig.get_size_inches()}")
                    print(f"   Number of axes: {len(fig.axes)}")
                else:
                    print(f"❌ {name} - Figure is None")
            except Exception as e:
                print(f"❌ {name} - Error: {e}")
        
        print(f"\n📈 Summary:")
        print(f"   Total figures created: {len(figures)}/5")
        print(f"   Expected: 5 figures")
        
        if len(figures) == 5:
            print("🎉 SUCCESS: All 5 outputs are working correctly!")
            return True
        else:
            print("⚠️  WARNING: Some outputs failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_all_outputs()
    if success:
        print("\n🚀 Ready to use in GUI!")
        print("   Run 'python main_app.py' to start the GUI")
        print("   Load a model and click 'Run Analysis' to see all 5 outputs")
    else:
        print("\n🔧 Please fix the issues before using the GUI") 