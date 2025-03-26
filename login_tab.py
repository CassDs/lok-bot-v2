import tkinter as tk
from tkinter import ttk, StringVar, messagebox
import sys
import threading
from iqoptionapi.stable_api import IQ_Option
import os
import pickle
from data_collection_tab import DataCollectionTab

class LoginTab:
    def __init__(self, master):
        self.frame = ttk.Frame(master)
        self._build_login_ui()
        self.iq = None
        self.connected = False
        self.credentials_file = "iq_credentials.pkl"
        self._load_saved_credentials()
        self.on_connected_callback = None 

    def set_on_connected_callback(self, callback):
        self.on_connected_callback = callback

    def _build_login_ui(self):
        login_frame = ttk.LabelFrame(self.frame, text="Credenciais IQ Option")
        login_frame.pack(fill="x", expand=False, padx=10, pady=10)

        # Email
        ttk.Label(login_frame, text="Email:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.email_var = StringVar()
        self.email_entry = ttk.Entry(login_frame, textvariable=self.email_var, width=40)
        self.email_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        # Senha
        ttk.Label(login_frame, text="Senha:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.password_var = StringVar()
        self.password_entry = ttk.Entry(login_frame, textvariable=self.password_var, width=40, show="*")
        self.password_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        # Tipo de conta
        ttk.Label(login_frame, text="Tipo de Conta:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.account_type_var = StringVar(value="demo")
        account_type_frame = ttk.Frame(login_frame)
        account_type_frame.grid(row=2, column=1, sticky="w", padx=5, pady=5)

        ttk.Radiobutton(account_type_frame, text="Demo", variable=self.account_type_var, value="demo").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(account_type_frame, text="Real", variable=self.account_type_var, value="real").pack(side=tk.LEFT, padx=10)

        # Checkbox lembrar
        self.remember_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(login_frame, text="Lembrar Credenciais", variable=self.remember_var).grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=5)

        # Botão de login
        self.login_button = ttk.Button(login_frame, text="Conectar", command=self.connect_to_iqoption)
        self.login_button.grid(row=4, column=0, columnspan=2, pady=10)

        # Status
        self.login_status_var = StringVar(value="Aguardando conexão...")
        ttk.Label(login_frame, textvariable=self.login_status_var).grid(row=5, column=0, columnspan=2, pady=5)

        # Barra de progresso
        self.login_progress = ttk.Progressbar(login_frame, orient="horizontal", mode="indeterminate")
        self.login_progress.grid(row=6, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        # Área de status
        connection_frame = ttk.LabelFrame(self.frame, text="Status da Conexão")
        connection_frame.pack(fill="x", expand=False, padx=10, pady=10)

        self.connection_info_text = tk.Text(connection_frame, height=8, width=80)
        self.connection_info_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.connection_info_text.config(state=tk.DISABLED)

    def connect_to_iqoption(self):
        self.login_progress.start()
        self.login_status_var.set("Conectando...")
        threading.Thread(target=self._connect_thread).start()

    def _connect_thread(self):
        email = self.email_var.get()
        password = self.password_var.get()
        account_type = self.account_type_var.get()

        # Mapeia para valores válidos da API
        balance_map = {
            "demo": "PRACTICE",
            "real": "REAL",
        }

        try:
            self.iq = IQ_Option(email, password)
            self.iq.connect()

            if self.iq.check_connect():
                if account_type in balance_map:
                    self.iq.change_balance(balance_map[account_type])

                balance_mode = self.iq.get_balance_mode()
                balance_amount = self.iq.get_balance()

                self.connected = True
                self.login_status_var.set("Conectado com sucesso!")
                self._log_connection_info(f"Conectado como: {email}\nTipo de conta: {account_type}\nModo atual: {balance_mode}\nSaldo: {balance_amount}")

                if self.remember_var.get():
                    self._save_credentials(email, password, account_type)
                    
                # Chama o callback se estiver definido
                if self.on_connected_callback:
                    self.on_connected_callback(self.iq)
            else:
                self.login_status_var.set("Falha na conexão")
                self._log_connection_info("Erro: Email ou senha incorretos")
        except Exception as e:
            self.login_status_var.set("Erro na conexão")
            self._log_connection_info(f"Erro ao conectar: {str(e)}")

        self.login_progress.stop()
        

    def _log_connection_info(self, text):
        self.connection_info_text.config(state=tk.NORMAL)
        self.connection_info_text.insert(tk.END, text + "\n")
        self.connection_info_text.see(tk.END)
        self.connection_info_text.config(state=tk.DISABLED)

    def _save_credentials(self, email, password, account_type):
        data = {"email": email, "password": password, "account_type": account_type}
        with open(self.credentials_file, "wb") as f:
            pickle.dump(data, f)

    def _load_saved_credentials(self):
        if os.path.exists(self.credentials_file):
            with open(self.credentials_file, "rb") as f:
                data = pickle.load(f)
                self.email_var.set(data.get("email", ""))
                self.password_var.set(data.get("password", ""))
                self.account_type_var.set(data.get("account_type", "demo"))