"""
🚀 AI IT SUPPORT TICKET PREDICTOR - PROFESSIONAL LAUNCHER
=========================================================
Choose your preferred interface to run the application
"""

import subprocess
import sys
import os

def print_banner():
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║    🤖 AI-POWERED IT SUPPORT TICKET PREDICTOR - PROFESSIONAL LAUNCHER 🤖     ║
    ║                                                                              ║
    ║         Revolutionary predictive analytics for IT support operations        ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)

def main():
    print_banner()
    
    print("📋 Available Applications:")
    print("1. 🖥️  Desktop Application (Tkinter GUI)")
    print("2. 📊 Professional Streamlit Dashboard")
    print("3. 🎯 Enhanced Analytics Dashboard")
    print("4. ❌ Exit")
    
    while True:
        try:
            choice = input("\n🎯 Select an application (1-4): ").strip()
            
            if choice == "1":
                print("\n🚀 Launching Desktop Application...")
                subprocess.run([sys.executable, "working_demo.py"])
                break
                
            elif choice == "2":
                print("\n🚀 Launching Streamlit Dashboard...")
                subprocess.run([sys.executable, "-m", "streamlit", "run", "simple_dashboard.py"])
                break
                
            elif choice == "3":
                print("\n🚀 Launching Enhanced Analytics Dashboard...")
                subprocess.run([sys.executable, "-m", "streamlit", "run", "enhanced_dashboard.py"])
                break
                
            elif choice == "4":
                print("\n👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()