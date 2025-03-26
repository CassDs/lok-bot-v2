import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import os

class DataProcessingTab:
    def __init__(self, master):
        self.frame = ttk.Frame(master)
        self.data = None
        self.cleaned_data = None
        self.data_collection_tab = None  # Referência para acessar os dados coletados
        self._build_ui()

    def set_data_collection_tab(self, data_collection_tab):
        """Permite acesso aos dados da aba de coleta."""
        self.data_collection_tab = data_collection_tab

    def _build_ui(self):
        # Frame para seleção da fonte de dados
        source_frame = ttk.LabelFrame(self.frame, text="Fonte de Dados")
        source_frame.pack(fill="x", padx=10, pady=10)

        # Opções para fonte de dados (botões de rádio)
        self.data_source = tk.StringVar(value="file")
        ttk.Radiobutton(source_frame, text="Carregar de arquivo", variable=self.data_source, 
                       value="file", command=self._update_buttons).pack(anchor="w", padx=10, pady=2)
        ttk.Radiobutton(source_frame, text="Usar dados da coleta atual", variable=self.data_source, 
                       value="collection", command=self._update_buttons).pack(anchor="w", padx=10, pady=2)

        # Frame para ações
        control_frame = ttk.LabelFrame(self.frame, text="Processamento de Dados")
        control_frame.pack(fill="x", padx=10, pady=10)

        # Botões
        self.load_csv_btn = ttk.Button(control_frame, text="Carregar CSV", command=self._carregar_csv)
        self.load_csv_btn.pack(side=tk.LEFT, padx=5)
        
        self.load_collection_btn = ttk.Button(control_frame, text="Carregar da Coleta", 
                                            command=self._carregar_da_coleta, state=tk.DISABLED)
        self.load_collection_btn.pack(side=tk.LEFT, padx=5)
        
        self.clean_btn = ttk.Button(control_frame, text="Aplicar Limpeza", 
                                  command=self._aplicar_limpeza, state=tk.DISABLED)
        self.clean_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(control_frame, text="Salvar Dados Processados", 
                                 command=self._salvar_processado, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar(value="Aguardando ação...")
        ttk.Label(self.frame, textvariable=self.status_var).pack(pady=5)

        preview_frame = ttk.LabelFrame(self.frame, text="Visualização de Dados")
        preview_frame.pack(fill="both", expand=True, padx=10, pady=10)

        columns = ("timestamp", "abertura", "alta", "baixa", "fechamento", "volume")
        self.tree = ttk.Treeview(preview_frame, columns=columns, show="headings")
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center", width=100)

        vsb = ttk.Scrollbar(preview_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)

    def _update_buttons(self):
        """Atualiza o estado dos botões com base na fonte de dados selecionada."""
        if self.data_source.get() == "file":
            self.load_csv_btn.config(state=tk.NORMAL)
            self.load_collection_btn.config(state=tk.DISABLED)
        else:  # collection
            self.load_csv_btn.config(state=tk.DISABLED)
            self.load_collection_btn.config(state=tk.NORMAL)
            
        # Reseta os estados dos outros botões
        self.clean_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)

    def _carregar_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            self.data = pd.read_csv(file_path)
            self.status_var.set(f"Dados carregados: {os.path.basename(file_path)}")
            self._atualizar_visualizacao(self.data)
            self.clean_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar CSV: {str(e)}")

    def _carregar_da_coleta(self):
        """Carrega dados da aba de coleta."""
        if not self.data_collection_tab or self.data_collection_tab.collected_data is None:
            messagebox.showwarning("Aviso", "Nenhum dado coletado disponível.")
            return
            
        try:
            self.data = self.data_collection_tab.collected_data.copy()
            self.status_var.set(f"Dados carregados da coleta: {len(self.data)} registros")
            self._atualizar_visualizacao(self.data)
            self.clean_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar dados da coleta: {str(e)}")

    def _aplicar_limpeza(self):
        if self.data is None:
            messagebox.showwarning("Aviso", "Nenhum dado carregado.")
            return

        try:
            df = self.data.copy()
            df = df.dropna()
            df = df.drop_duplicates()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values("timestamp")
            self.cleaned_data = df
            self.status_var.set("Dados limpos com sucesso!")
            self._atualizar_visualizacao(df)
            self.save_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Erro", f"Erro no processamento: {str(e)}")

    def _salvar_processado(self):
        if self.cleaned_data is None:
            messagebox.showwarning("Aviso", "Nenhum dado processado para salvar.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.cleaned_data.to_csv(file_path, index=False)
            self.status_var.set(f"Dados processados salvos em: {file_path}")

    def _atualizar_visualizacao(self, df):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for _, row in df.head(100).iterrows():  # Mostra só os primeiros 100
            self.tree.insert("", "end", values=tuple(row))
