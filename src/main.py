from gui.dataset_CLI import run_datasetCLI
from gui.train_CLI import run_trainCLI
from gui.predict_CLI import run_predictCLI

class CLIApplication:
    def __init__(self):
        pass

    def run(self):
        while True:
            self.show_menu()
            choice = input("Select an option (1-3): ").strip()

            if choice == '1':
                self.execute_command(run_datasetCLI())
            elif choice == '2':
                self.execute_command(run_trainCLI())
            elif choice == '3':
                self.execute_command(run_predictCLI())
            elif choice == 'q':
                print("Exiting...")
                break
            else:
                print("Invalid option, please choose again.")

    def show_menu(self):
        print("\nML Pipeline CLI")
        print("1. Dataset Creation and Editing")
        print("2. Train Model")
        print("3. Predict Image")
        print("q. Exit")

    def execute_command(self, command_func):
        try:
            command_func()
        except Exception as e:
            print(f"Error occurred: {e}")

if __name__ == "__main__":
    app = CLIApplication()
    app.run()
