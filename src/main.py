"""
is_miko: main.py

A command-line interface for managing a machine learning pipeline.
This application allows users to create and edit datasets, train models,
and make predictions.

Modules:
- Dataset CLI: Handles dataset creation and editing.
- Train CLI: Manages model training.
- Predict CLI: Facilitates image predictions.

Author: Matthew Guo
Date: 24.09.09
"""

from gui.dataset_CLI import run_datasetCLI
from gui.train_CLI import run_trainCLI
from gui.predict_CLI import run_predictCLI

class CLIApplication:
    def __init__(self):
        """Initialize the CLI application."""
        pass

    def run(self):
        """Run the main application loop."""
        while True:
            self.show_menu()
            choice = input("Select an option (1-3, or 'q' to quit): ").strip()

            if choice == '1':
                self.execute_command(run_datasetCLI)
            elif choice == '2':
                self.execute_command(run_trainCLI)
            elif choice == '3':
                self.execute_command(run_predictCLI)
            elif choice == 'q':
                print("Exiting...")
                break
            else:
                print("Invalid option, please choose again.")

    def show_menu(self):
        """Display the main menu options."""
        print("\nML Pipeline CLI")
        print("1. Dataset Creation and Editing")
        print("2. Train Model")
        print("3. Predict Image")
        print("q. Exit")

    def execute_command(self, command_func):
        """Execute the given command function with error handling."""
        try:
            command_func()
        except Exception as e:
            print(f"Error occurred: {e}")

if __name__ == "__main__":
    app = CLIApplication()
    app.run()
