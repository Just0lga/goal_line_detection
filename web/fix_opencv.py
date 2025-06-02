#!/usr/bin/env python3
"""
OpenCV Installation Fixer
This script helps diagnose and fix OpenCV installation issues.
"""

import sys
import subprocess
import os

def run_command(cmd):
    """Run a command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_opencv():
    """Check current OpenCV installation"""
    print("🔍 Checking current OpenCV installation...")
    
    try:
        import cv2
        print(f"✅ OpenCV is installed! Version: {cv2.__version__}")
        return True
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False

def list_opencv_packages():
    """List all OpenCV-related packages"""
    print("\n📦 Checking installed OpenCV packages...")
    success, stdout, stderr = run_command("pip list | grep opencv")
    if success and stdout:
        print("Found OpenCV packages:")
        print(stdout)
    else:
        print("❌ No OpenCV packages found")

def fix_opencv():
    """Fix OpenCV installation"""
    print("\n🔧 Fixing OpenCV installation...")
    
    # Step 1: Uninstall all OpenCV packages
    print("Step 1: Removing existing OpenCV packages...")
    opencv_packages = ['opencv-python', 'opencv-python-headless', 'opencv-contrib-python']
    
    for package in opencv_packages:
        print(f"  Uninstalling {package}...")
        run_command(f"pip uninstall {package} -y")
    
    # Step 2: Install opencv-python-headless
    print("Step 2: Installing opencv-python-headless...")
    success, stdout, stderr = run_command("pip install opencv-python-headless==4.8.1.78")
    
    if success:
        print("✅ opencv-python-headless installed successfully!")
    else:
        print(f"❌ Installation failed: {stderr}")
        return False
    
    # Step 3: Test import
    print("Step 3: Testing OpenCV import...")
    return check_opencv()

def main():
    print("🛠️  OpenCV Installation Fixer")
    print("=" * 40)
    
    # Check current installation
    if check_opencv():
        print("\n✅ OpenCV is working correctly! No fix needed.")
        return
    
    # List current packages
    list_opencv_packages()
    
    # Ask user if they want to fix
    response = input("\n❓ Do you want to fix the OpenCV installation? (y/n): ").lower()
    if response != 'y':
        print("🚫 Fix cancelled.")
        return
    
    # Attempt fix
    if fix_opencv():
        print("\n🎉 OpenCV installation fixed successfully!")
        print("✅ You can now run your Flask app.")
    else:
        print("\n😞 Failed to fix OpenCV installation.")
        print("💡 Manual steps:")
        print("   1. pip uninstall opencv-python opencv-python-headless")
        print("   2. pip install opencv-python-headless==4.8.1.78")
        print("   3. Restart your terminal/IDE")

if __name__ == "__main__":
    main() 