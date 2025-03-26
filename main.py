import tkinter as tk
from tkinter import ttk
from login_tab import LoginTab
from data_collection_tab import DataCollectionTab
from processing_tab import DataProcessingTab

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

        # Coleta de Dados
        self.data_collection_tab = DataCollectionTab(self.notebook)
        self.notebook.add(self.data_collection_tab.frame, text="Coleta de Dados")

        # Processamento de Dados
        self.processing_tab = DataProcessingTab(self.notebook)
        self.notebook.add(self.processing_tab.frame, text="Processamento de Dados")
        
        # Conectar a aba de processamento com a aba de coleta
        self.processing_tab.set_data_collection_tab(self.data_collection_tab)
        
        # Configurar o callback de conexão
        self._setup_connection_callbacks()

    def _setup_connection_callbacks(self):
        # Método para ser chamado após uma conexão bem-sucedida
        self.login_tab.set_on_connected_callback(self._on_iq_connected)
        
    def _on_iq_connected(self, iq_instance):
        self.data_collection_tab.set_iq_option_connection(iq_instance)
        # Muda para a aba de coleta de dados após o login
        self.notebook.select(1)  # índice 1 = segunda aba

if __name__ == "__main__":
    root = tk.Tk()
    app = IQOptionApp(root)
    root.mainloop()