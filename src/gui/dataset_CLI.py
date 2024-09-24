"""
dataset_CLI.py

A command-line interface for managing a dataset of videos.
This application allows users to add, remove, and list video entries
along with their corresponding watch time and total time.

Author: Matthew Guo
Date: 24.09.09
"""

import pandas as pd
import os
import shutil
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter

class DatasetCLI:
    def __init__(self):
        """Initialize the dataset management CLI."""
        self.filename = 'dataset.csv'
        if os.path.exists(self.filename):
            self.df = pd.read_csv(self.filename)
        else:
            self.df = pd.DataFrame(columns=['Video Path', 'Watch Time', 'Total Time'])
    
    def list_entries(self):
        """List all entries in the dataset."""
        if self.df.empty:
            print("No entries found.")
        else:
            print(self.df.to_string(index=False))
    
    def add_entry(self):
        """Add a new entry to the dataset."""
        video_path = self.prompt_for_path("Enter video path: ")
        watch_time = input("Enter watch time: ").strip()
        total_time = input("Enter total time: ").strip()

        if not (video_path and watch_time and total_time):
            print("Error: All fields must be filled")
            return

        video_data_dir = os.path.join(os.getcwd(), 'videoData')
        if not os.path.exists(video_data_dir):
            os.makedirs(video_data_dir)

        video_filename = os.path.basename(video_path)
        new_video_path = os.path.join(video_data_dir, video_filename)

        try:
            shutil.move(video_path, new_video_path)
        except (FileNotFoundError, PermissionError) as e:
            print(f"Error: {e}")
            return
        except Exception as e:
            print(f"Error: An unexpected error occurred: {e}")
            return

        relative_video_path = os.path.relpath(new_video_path, os.getcwd())
        new_row_df = pd.DataFrame([{
            'Video Path': relative_video_path,
            'Watch Time': watch_time,
            'Total Time': total_time
        }])

        self.df = pd.concat([self.df, new_row_df], ignore_index=True)
        self.df.to_csv(self.filename, index=False)
        print("Entry added successfully.")
    
    def remove_entry(self):
        """Remove an entry from the dataset."""
        video_path = self.prompt_for_path("Enter video path to remove: ")
        if video_path in self.df['Video Path'].values:
            self.df = self.df[self.df['Video Path'] != video_path]
            self.df.to_csv(self.filename, index=False)
            print("Entry removed successfully.")
        else:
            print("Error: Video path not found.")
    
    def prompt_for_path(self, prompt_message):
        """Prompt for a file or directory path with tab completion."""
        path_completer = PathCompleter()
        return prompt(prompt_message, completer=path_completer).strip()

def run_datasetCLI():
    """Run the dataset management command-line interface."""
    cli = DatasetCLI()
    while True:
        print("\nDataset Management CLI")
        print("1. List entries")
        print("2. Add entry")
        print("3. Remove entry")
        print("q. Exit")
        choice = input("Select an option (1-4): ").strip()

        if choice == '1':
            cli.list_entries()
        elif choice == '2':
            cli.add_entry()
        elif choice == '3':
            cli.remove_entry()
        elif choice == 'q':
            break
        else:
            print("Invalid option, please choose again.")
