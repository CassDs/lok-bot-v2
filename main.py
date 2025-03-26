import tkinter as tk
from tkinter import ttk
from login_tab import LoginTab

class IQOptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("IQ Option - Coletor de Dados")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except:
            pass

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=1, fill="both", padx=5, pady=5)

        # Inicializar abas
        self._init_tabs()

    def _init_tabs(self):
        # Login
        self.login_tab = LoginTab(self.notebook)
        self.notebook.add(self.login_tab.frame, text="Login IQ Option")

if __name__ == "__main__":
    root = tk.Tk()
    app = IQOptionApp(root)
    root.mainloop()
