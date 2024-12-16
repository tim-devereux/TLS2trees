import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from plyfile import PlyData, PlyElement, PlyProperty
import os

class PlyChannelManager:
    def __init__(self, root):
        self.root = root
        self.root.title("PLY Channel Manager")
        self.root.geometry("600x500")
        
        self.current_ply = None
        self.selected_channel = tk.StringVar()
        
        self.setup_ui()
        
    def setup_ui(self):
        # File selection frame
        file_frame = ttk.LabelFrame(self.root, text="File Selection", padding=10)
        file_frame.pack(fill="x", padx=10, pady=5)
        
        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path, width=50).pack(side="left", padx=5)
        ttk.Button(file_frame, text="Browse", command=self.load_ply).pack(side="left")
        
        # Channel list frame
        channel_frame = ttk.LabelFrame(self.root, text="Channels", padding=10)
        channel_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Channel listbox
        self.channel_listbox = tk.Listbox(channel_frame, width=40, height=10)
        self.channel_listbox.pack(fill="both", expand=True)
        self.channel_listbox.bind('<<ListboxSelect>>', self.on_select_channel)
        
        # Operations frame
        ops_frame = ttk.LabelFrame(self.root, text="Operations", padding=10)
        ops_frame.pack(fill="x", padx=10, pady=5)
        
        # Rename frame
        rename_frame = ttk.Frame(ops_frame)
        rename_frame.pack(fill="x", pady=5)
        ttk.Label(rename_frame, text="New name:").pack(side="left")
        self.new_name = tk.StringVar()
        ttk.Entry(rename_frame, textvariable=self.new_name, width=30).pack(side="left", padx=5)
        ttk.Button(rename_frame, text="Rename", command=self.rename_channel).pack(side="left")
        
        # Remove button
        ttk.Button(ops_frame, text="Remove Channel", command=self.remove_channel).pack(pady=5)
        
        # Save frame
        save_frame = ttk.Frame(self.root)
        save_frame.pack(fill="x", padx=10, pady=5)
        ttk.Button(save_frame, text="Save PLY", command=self.save_ply).pack(side="right")
    
    def load_ply(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("PLY files", "*.ply"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.current_ply = PlyData.read(filepath)
                self.file_path.set(filepath)
                self.update_channel_list()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load PLY file: {str(e)}")
    
    def update_channel_list(self):
        self.channel_listbox.delete(0, tk.END)
        if self.current_ply:
            for prop in self.current_ply['vertex'].properties:
                self.channel_listbox.insert(tk.END, prop.name)
    
    def on_select_channel(self, event):
        if self.channel_listbox.curselection():
            self.selected_channel.set(self.channel_listbox.get(self.channel_listbox.curselection()))
    
    def rename_channel(self):
        if not self.current_ply:
            messagebox.showwarning("Warning", "Please load a PLY file first")
            return
            
        selected_idx = self.channel_listbox.curselection()
        if not selected_idx:
            messagebox.showwarning("Warning", "Please select a channel to rename")
            return
            
        old_name = self.channel_listbox.get(selected_idx)
        new_name = self.new_name.get().strip()
        
        if not new_name:
            messagebox.showwarning("Warning", "Please enter a new name")
            return
            
        try:
            vertex_data = self.current_ply['vertex']
            data = vertex_data.data
            
            # Get the old data
            old_dtype = data.dtype.descr
            new_dtype = []
            
            # Create new dtype with renamed field
            for dt in old_dtype:
                if dt[0] == old_name:
                    new_dtype.append((new_name, dt[1]))
                else:
                    new_dtype.append(dt)
            
            # Create new array with renamed field
            new_data = np.empty(len(data), dtype=new_dtype)
            
            # Copy data to new array
            for dt in old_dtype:
                name = dt[0]
                if name == old_name:
                    new_data[new_name] = data[old_name]
                else:
                    new_data[name] = data[name]
            
            # Create new vertex element
            new_vertex = PlyElement.describe(new_data, 'vertex')
            self.current_ply = PlyData([new_vertex], text=self.current_ply.text)
            
            self.update_channel_list()
            messagebox.showinfo("Success", f"Renamed channel '{old_name}' to '{new_name}'")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to rename channel: {str(e)}")
    
    def remove_channel(self):
        if not self.current_ply:
            messagebox.showwarning("Warning", "Please load a PLY file first")
            return
            
        selected_idx = self.channel_listbox.curselection()
        if not selected_idx:
            messagebox.showwarning("Warning", "Please select a channel to remove")
            return
            
        channel_name = self.channel_listbox.get(selected_idx)
        
        if messagebox.askyesno("Confirm", f"Are you sure you want to remove channel '{channel_name}'?"):
            try:
                vertex_data = self.current_ply['vertex']
                data = vertex_data.data
                
                # Get current dtype without the field to remove
                old_dtype = data.dtype.descr
                new_dtype = [dt for dt in old_dtype if dt[0] != channel_name]
                
                if len(new_dtype) == len(old_dtype):
                    messagebox.showwarning("Warning", "Channel not found")
                    return
                
                # Create new array without the removed field
                new_data = np.empty(len(data), dtype=new_dtype)
                
                # Copy remaining fields
                for dt in new_dtype:
                    name = dt[0]
                    new_data[name] = data[name]
                
                # Create new vertex element
                new_vertex = PlyElement.describe(new_data, 'vertex')
                self.current_ply = PlyData([new_vertex], text=self.current_ply.text)
                
                self.update_channel_list()
                messagebox.showinfo("Success", f"Removed channel '{channel_name}'")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to remove channel: {str(e)}")
    
    def save_ply(self):
        if not self.current_ply:
            messagebox.showwarning("Warning", "Please load a PLY file first")
            return
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=".ply",
            filetypes=[("PLY files", "*.ply"), ("All files", "*.*")],
            initialdir=os.path.dirname(self.file_path.get()),
            initialfile=f"modified_{os.path.basename(self.file_path.get())}"
        )
        
        if filepath:
            try:
                self.current_ply.write(filepath)
                messagebox.showinfo("Success", "File saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PlyChannelManager(root)
    root.mainloop()