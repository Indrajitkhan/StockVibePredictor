#!/usr/bin/env python3
"""
Script to run the Universal Stock Training System
This will train models for popular stocks including NIFTY and international markets
"""

import os
import sys
import time
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

# Import our universal training system
from UniversalTrainModel import batch_train_popular_stocks, train_universal_model, logger

def main():
    """Main function to run the training system"""
    print("🚀 Starting Universal Stock Training System...")
    print("=" * 60)
    
    # Create necessary directories
    base_dir = current_dir.parent
    models_dir = base_dir / "Models"
    logs_dir = base_dir / "Logs"
    
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    
    print(f"📁 Models will be saved to: {models_dir}")
    print(f"📋 Logs will be saved to: {logs_dir}")
    print()
    
    try:
        # Record start time
        start_time = time.time()
        
        # Run batch training for popular stocks
        print("🎯 Training models for popular stocks...")
        successful, failed = batch_train_popular_stocks()
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETED!")
        print("=" * 60)
        print(f"⏱️  Total time: {minutes}m {seconds}s")
        print(f"✅ Successful models: {successful}")
        print(f"❌ Failed models: {failed}")
        print(f"📊 Success rate: {successful/(successful+failed)*100:.1f}%")
        print()
        
        # List created models
        model_files = list(models_dir.glob("*_model.pkl"))
        if model_files:
            print("🎉 Models created:")
            for model_file in sorted(model_files):
                ticker = model_file.stem.replace("_model", "").replace("INDEX_", "^").replace("_", ".")
                print(f"   📈 {ticker}")
            print()
        
        # Check if NIFTY model was created
        nifty_model = models_dir / "INDEX_NSEI_model.pkl"
        if nifty_model.exists():
            print("🇮🇳 NIFTY model successfully created! Now you can search for 'NIFTY' in the frontend.")
        else:
            print("⚠️  NIFTY model was not created. You may need to check the training logs.")
        
        print()
        print("🌟 Your backend now supports predictions for ALL stocks in your frontend database!")
        print("🔍 Users can now search for any stock (including NIFTY) and get predictions.")
        print("🤖 Models will be trained automatically for new stocks on first request.")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        logger.error(f"Training script error: {str(e)}")
    
    print("\n🏁 Training script finished!")

if __name__ == "__main__":
    main()
