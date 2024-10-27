import matplotlib.pyplot as plt
import pandas as pd
import time
import argparse
from pathlib import Path
import matplotlib.animation as animation
from matplotlib.lines import Line2D

class TrainingMonitor:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.last_modified = self.file_path.stat().st_mtime
        
        # Set up the figure and subplots
        plt.style.use('seaborn')
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Initialize empty lines
        self.lines = {
            'loss': Line2D([], [], color='blue', label='Training Loss', marker='o'),
            'val_loss': Line2D([], [], color='red', label='Validation Loss', marker='o'),
            'acc': Line2D([], [], color='green', label='Training Accuracy', marker='o'),
            'val_acc': Line2D([], [], color='purple', label='Validation Accuracy', marker='o')
        }
        
        # Add lines to axes
        self.ax1.add_line(self.lines['loss'])
        self.ax1.add_line(self.lines['val_loss'])
        self.ax2.add_line(self.lines['acc'])
        self.ax2.add_line(self.lines['val_acc'])
        
        # Configure axes
        self.ax1.set_ylabel('Loss')
        self.ax1.grid(True)
        self.ax1.legend()
        
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy')
        self.ax2.grid(True)
        self.ax2.legend()
        
        # Initial data load
        self.update_data()
        
    def update_data(self):
        try:
            df = pd.read_csv(self.file_path)
            
            # Update lines data
            for line_name, line in self.lines.items():
                if line_name == 'loss':
                    x, y = df['epoch'], df['epoch_loss']
                elif line_name == 'val_loss':
                    x, y = df['epoch'], df['val_epoch_loss']
                elif line_name == 'acc':
                    x, y = df['epoch'], df['epoch_acc']
                else:  # val_acc
                    x, y = df['epoch'], df['val_epoch_acc']
                
                line.set_data(x, y)
            
            # Update axis limits
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax2.relim()
            self.ax2.autoscale_view()
            
            # Add some padding to the y-axis
            for ax in [self.ax1, self.ax2]:
                ymin, ymax = ax.get_ylim()
                padding = (ymax - ymin) * 0.1
                ax.set_ylim(ymin - padding, ymax + padding)
            
            return True
        except Exception as e:
            print(f"Error updating data: {e}")
            return False

    def animate(self, frame):
        current_modified = self.file_path.stat().st_mtime
        if current_modified != self.last_modified:
            self.last_modified = current_modified
            self.update_data()
        return list(self.lines.values())

def main():
    parser = argparse.ArgumentParser(description='Monitor training metrics in real-time')
    parser.add_argument('file_path', type=str, help='Path to the CSV file containing training metrics')
    parser.add_argument('--interval', type=int, default=1000, 
                       help='Update interval in milliseconds (default: 1000)')
    args = parser.parse_args()
    
    # Verify file exists and has correct format
    file_path = Path(args.file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        df = pd.read_csv(file_path)
        required_columns = ['epoch', 'epoch_loss', 'epoch_acc', 'val_epoch_loss', 'val_epoch_acc']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    except pd.errors.EmptyDataError:
        raise ValueError("File is empty")
    except pd.errors.ParserError:
        raise ValueError("Invalid CSV file format")
    
    # Create and run the monitor
    monitor = TrainingMonitor(file_path)
    ani = animation.FuncAnimation(
        monitor.fig, 
        monitor.animate, 
        interval=args.interval,
        blit=True
    )
    
    plt.show()

if __name__ == "__main__":
    main()