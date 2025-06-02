#!/usr/bin/env python3
"""
Goal Line Detection Web Application Startup Script

This script helps you start the web application with proper environment setup
and provides helpful information about the system status.
"""

import os
import sys
import subprocess
import platform
import socket
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        ('flask', 'flask'), 
        ('opencv', 'cv2'),  # Check for cv2 module instead of package name
        ('numpy', 'numpy'), 
        ('torch', 'torch'), 
        ('ultralytics', 'ultralytics')
    ]
    
    optional_packages = [('detectron2', 'detectron2')]
    
    missing_required = []
    missing_optional = []
    
    for display_name, module_name in required_packages:
        try:
            __import__(module_name)
            print(f"✅ {display_name}")
        except ImportError:
            missing_required.append(display_name)
            print(f"❌ {display_name}")
    
    for display_name, module_name in optional_packages:
        try:
            __import__(module_name)
            print(f"✅ {display_name} (optional)")
        except ImportError:
            missing_optional.append(display_name)
            print(f"⚠️  {display_name} (optional)")
    
    return missing_required, missing_optional

def check_models():
    """Check if trained models are available"""
    models = {
        'YOLO (Custom Trained)': '../goal-seg/exp3050ti2/weights/best.pt',
        'YOLO (Alternative)': '../yolo11n.pt',
        'Detectron2': '../output/model_final.pth'
    }
    
    available_models = []
    
    for name, path in models.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"✅ {name} model found ({size:.1f}MB): {path}")
            available_models.append(name)
        else:
            print(f"⚠️  {name} model not found: {path}")
    
    return available_models

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available: {device_count} GPU(s)")
            print(f"   Primary GPU: {device_name}")
            return True
        else:
            print("⚠️  CUDA not available (CPU only)")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def get_local_ip():
    """Get local IP address"""
    try:
        # Connect to a remote address to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "localhost"

def install_dependencies(packages):
    """Install missing packages"""
    if not packages:
        return True
    
    # Map display names to actual pip package names
    package_map = {
        'flask': 'flask',
        'opencv': 'opencv-python-headless==4.8.1.78',
        'numpy': 'numpy',
        'torch': 'torch',
        'ultralytics': 'ultralytics',
        'detectron2': 'detectron2'
    }
    
    pip_packages = [package_map.get(pkg, pkg) for pkg in packages]
    
    print(f"\n📦 Installing missing packages: {', '.join(pip_packages)}")
    
    try:
        cmd = [sys.executable, '-m', 'pip', 'install'] + pip_packages
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print(f"   Error output: {e.stderr}")
        return False

def main():
    """Main startup function"""
    print("🚀 Goal Line Detection Web Application")
    print("=" * 50)
    
    # Check system requirements
    print("\n🔍 Checking System Requirements:")
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    print("\n📋 Checking Dependencies:")
    missing_required, missing_optional = check_dependencies()
    
    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        response = input("Would you like to install them? (y/n): ").lower().strip()
        if response == 'y':
            if not install_dependencies(missing_required):
                sys.exit(1)
        else:
            print("Cannot continue without required dependencies")
            sys.exit(1)
    
    # Check models
    print("\n🤖 Checking AI Models:")
    available_models = check_models()
    
    if not available_models:
        print("\n⚠️  No trained models found!")
        print("   The application will use pre-trained models with limited accuracy.")
        print("   For best results, place your trained models in the correct locations:")
        print("   - YOLO: ../yolo11n.pt")
        print("   - Detectron2: ../output/model_final.pth")
    
    # Check CUDA
    print("\n🔧 Checking GPU Support:")
    cuda_available = check_cuda()
    
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    print("\n📁 Created necessary directories")
    
    # Get network info
    local_ip = get_local_ip()
    
    # Start the application
    print("\n🌐 Starting Web Application...")
    print("=" * 50)
    print("🎯 Goal Line Detection is ready!")
    print(f"📍 Local access: http://localhost:5000")
    print(f"🌍 Network access: http://{local_ip}:5000")
    print(f"🤖 Available models: {', '.join(available_models) if available_models else 'Pre-trained only'}")
    print(f"⚡ GPU acceleration: {'Enabled' if cuda_available else 'Disabled'}")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    # Change to the web directory if needed
    if not os.path.exists('app.py'):
        web_dir = Path(__file__).parent
        os.chdir(web_dir)
    
    # Start Flask app
    try:
        from app import app
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 