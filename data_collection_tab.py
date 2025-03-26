import tkinter as tk
from tkinter import ttk, StringVar, messagebox, filedialog
import threading
import time
import pandas as pd
from iqoptionapi.stable_api import IQ_Option


class DataCollectionTab:
    def __init__(self, master):
        self.frame = ttk.Frame(master)
        self._build_ui()
        self.iq = None
        self.collected_data = None

    def set_iq_option_connection(self, iq):
        self.iq = iq
        self.collect_button.config(state=tk.NORMAL)
        threading.Thread(target=lambda: self._load_assets()).start()

    def _load_assets(self, asset_type=None):
        if asset_type is None:
            asset_type = self.asset_type_var.get()
            
        try:
            self.status_var.set(f"Carregando ativos do tipo {asset_type}...")
            assets = self.iq.get_all_open_time()
            
            # Verificar se o tipo de ativo existe no dicionário retornado
            if asset_type not in assets:
                self.status_var.set(f"Tipo de ativo '{asset_type}' não disponível.")
                return
            
            # Obter lista de ativos do tipo selecionado que estão abertos
            ativos = [name for name, val in assets[asset_type].items() if val['open']]
            
            # Ordenar ativos em ordem alfabética
            ativos = sorted(ativos)
            
            # Atualizar o combobox com os ativos ordenados
            self.asset_combobox['values'] = ativos
            
            if ativos:
                self.asset_var.set(ativos[0])
                
            # Atualizar status com a contagem de ativos
            num_ativos = len(ativos)
            self.status_var.set(f"{num_ativos} ativos {asset_type} disponíveis")
            
        except Exception as e:
            self.status_var.set(f"Erro ao carregar ativos: {str(e)}")

    def _build_ui(self):
        input_frame = ttk.LabelFrame(self.frame, text="Parâmetros de Coleta de Dados")
        input_frame.pack(fill="x", expand=False, padx=10, pady=10)

        ttk.Label(input_frame, text="Ativo:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.asset_var = StringVar()
        self.asset_combobox = ttk.Combobox(input_frame, textvariable=self.asset_var, width=15)
        self.asset_combobox.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(input_frame, text="Tipo de Ativo:").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        self.asset_type_var = StringVar(value="binary")
        self.asset_type_menu = ttk.Combobox(input_frame, textvariable=self.asset_type_var, values=["binary", "digital", "forex", "crypto"], width=10)
        self.asset_type_menu.grid(row=0, column=3, sticky="w", padx=5, pady=5)
        self.asset_type_menu.bind("<<ComboboxSelected>>", lambda event: threading.Thread(target=lambda: self._load_assets()).start())

        ttk.Label(input_frame, text="Período:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.timeframe_var = StringVar(value="60")
        self.timeframe_menu = ttk.Combobox(input_frame, textvariable=self.timeframe_var, values=["60", "300", "900", "1800", "3600", "86400"], width=10)
        self.timeframe_menu.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(input_frame, text="Quantidade de Velas:").grid(row=1, column=2, sticky="w", padx=5, pady=5)
        self.candles_var = StringVar(value="1000")
        self.candles_entry = ttk.Entry(input_frame, textvariable=self.candles_var, width=10)
        self.candles_entry.grid(row=1, column=3, sticky="w", padx=5, pady=5)

        indicators_frame = ttk.LabelFrame(input_frame, text="Indicadores")
        indicators_frame.grid(row=2, column=0, columnspan=4, sticky="w", padx=5, pady=10)

        self.sma_var = tk.BooleanVar(value=True)
        self.ema_var = tk.BooleanVar(value=True)
        self.rsi_var = tk.BooleanVar(value=True)
        self.macd_var = tk.BooleanVar(value=True)
        self.bb_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(indicators_frame, text="SMA", variable=self.sma_var).pack(side=tk.LEFT)
        ttk.Checkbutton(indicators_frame, text="EMA", variable=self.ema_var).pack(side=tk.LEFT)
        ttk.Checkbutton(indicators_frame, text="RSI", variable=self.rsi_var).pack(side=tk.LEFT)
        ttk.Checkbutton(indicators_frame, text="MACD", variable=self.macd_var).pack(side=tk.LEFT)
        ttk.Checkbutton(indicators_frame, text="BB", variable=self.bb_var).pack(side=tk.LEFT)

        buttons_frame = ttk.Frame(input_frame)
        buttons_frame.grid(row=3, column=0, columnspan=4, pady=10)

        self.collect_button = ttk.Button(buttons_frame, text="Coletar Dados", command=self._start_data_collection, state=tk.DISABLED)
        self.collect_button.pack(side=tk.LEFT, padx=5)

        self.save_button = ttk.Button(buttons_frame, text="Salvar Dados", state=tk.DISABLED, command=self._salvar_dados)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = ttk.Button(buttons_frame, text="Limpar", command=self._limpar_campos)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.progress = ttk.Progressbar(self.frame, orient="horizontal", mode="indeterminate")
        self.progress.pack(fill="x", padx=10, pady=5)

        self.status_var = StringVar(value="Pronto para coletar dados")
        ttk.Label(self.frame, textvariable=self.status_var).pack(padx=10, pady=5)

        # Área de visualização dos dados
        data_view_frame = ttk.LabelFrame(self.frame, text="Visualização dos Dados")
        data_view_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Criar o Treeview para exibir os dados
        columns = ('timestamp', 'abertura', 'alta', 'baixa', 'fechamento', 'volume')
        self.data_tree = ttk.Treeview(data_view_frame, columns=columns, show='headings', height=10)
        
        # Definir os cabeçalhos
        self.data_tree.heading('timestamp', text='Data/Hora')
        self.data_tree.heading('abertura', text='Abertura')
        self.data_tree.heading('alta', text='Alta')
        self.data_tree.heading('baixa', text='Baixa')
        self.data_tree.heading('fechamento', text='Fechamento')
        self.data_tree.heading('volume', text='Volume')
        
        # Definir larguras das colunas
        self.data_tree.column('timestamp', width=150)
        self.data_tree.column('abertura', width=80)
        self.data_tree.column('alta', width=80)
        self.data_tree.column('baixa', width=80)
        self.data_tree.column('fechamento', width=80)
        self.data_tree.column('volume', width=80)
        
        # Adicionar barra de rolagem
        scrollbar = ttk.Scrollbar(data_view_frame, orient="vertical", command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=scrollbar.set)
        
        # Posicionar os elementos
        self.data_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _start_data_collection(self):
        threading.Thread(target=self._collect_data).start()

    def _collect_data(self):
        asset = self.asset_var.get()
        timeframe = int(self.timeframe_var.get())
        candles = int(self.candles_var.get())

        if not self.iq:
            self.status_var.set("Erro: Não conectado à IQ Option.")
            return

        self.progress.start()
        self.status_var.set("Coletando dados...")

        try:
            end_time = int(time.time())
            data = self.iq.get_candles(asset, timeframe, candles, end_time)

            if not data:
                raise Exception("Nenhum dado retornado pela API.")

            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['from'], unit='s')
            df = df.rename(columns={
                'open': 'abertura',
                'close': 'fechamento',
                'min': 'baixa',
                'max': 'alta',
                'volume': 'volume'
            })
            df = df[['timestamp', 'abertura', 'alta', 'baixa', 'fechamento', 'volume']]
            
            # Calcular indicadores técnicos selecionados
            self.status_var.set("Calculando indicadores técnicos...")
            
            if self.sma_var.get():
                # SMA de 20 períodos
                df['SMA_20'] = df['fechamento'].rolling(window=20).mean()
                # SMA de 50 períodos
                df['SMA_50'] = df['fechamento'].rolling(window=50).mean()
            
            if self.ema_var.get():
                # EMA de 20 períodos
                df['EMA_20'] = df['fechamento'].ewm(span=20, adjust=False).mean()
                # EMA de 50 períodos
                df['EMA_50'] = df['fechamento'].ewm(span=50, adjust=False).mean()
            
            if self.rsi_var.get():
                # Cálculo do RSI com período de 14
                delta = df['fechamento'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                df['RSI_14'] = 100 - (100 / (1 + rs))
            
            if self.macd_var.get():
                # MACD com períodos 12, 26
                ema12 = df['fechamento'].ewm(span=12, adjust=False).mean()
                ema26 = df['fechamento'].ewm(span=26, adjust=False).mean()
                df['MACD'] = ema12 - ema26
                df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            if self.bb_var.get():
                # Bandas de Bollinger com período de 20 e 2 desvios padrão
                sma = df['fechamento'].rolling(window=20).mean()
                std = df['fechamento'].rolling(window=20).std()
                df['BB_Superior'] = sma + (std * 2)
                df['BB_Medio'] = sma
                df['BB_Inferior'] = sma - (std * 2)

            self.collected_data = df
            self.status_var.set(f"Dados coletados com sucesso! Total de {len(df)} registros.")
            self.save_button.config(state=tk.NORMAL)
            
            # Limpar a tabela atual
            for i in self.data_tree.get_children():
                self.data_tree.delete(i)
            
            # Atualizar as colunas do Treeview para incluir os indicadores
            self._update_treeview_columns()
                    
            # Mostrar apenas os últimos 100 registros para não sobrecarregar a interface
            display_df = df.tail(100) if len(df) > 100 else df
            
            # Adicionar os dados à tabela
            for idx, row in display_df.iterrows():
                values = [
                    row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    f"{row['abertura']:.5f}",
                    f"{row['alta']:.5f}", 
                    f"{row['baixa']:.5f}",
                    f"{row['fechamento']:.5f}",
                    f"{row['volume']:.2f}"
                ]
                
                # Adicionar colunas de indicadores quando disponíveis
                for col in self.data_tree['columns'][6:]:  # Pular as 6 colunas básicas
                    if col in row and not pd.isna(row[col]):
                        values.append(f"{row[col]:.5f}")
                    else:
                        values.append("N/A")
                        
                self.data_tree.insert('', 'end', values=values)
                    
            # Rolar para o registro mais recente
            if self.data_tree.get_children():
                self.data_tree.see(self.data_tree.get_children()[-1])

        except Exception as e:
            self.status_var.set(f"Erro: {str(e)}")
            import traceback
            print(traceback.format_exc())

        self.progress.stop()

    def _update_treeview_columns(self):
        """Atualiza as colunas do Treeview baseado nos indicadores selecionados"""
        # Colunas básicas
        columns = ['timestamp', 'abertura', 'alta', 'baixa', 'fechamento', 'volume']
        
        # Adicionar colunas de indicadores
        if self.sma_var.get():
            columns.extend(['SMA_20', 'SMA_50'])
        if self.ema_var.get():
            columns.extend(['EMA_20', 'EMA_50'])
        if self.rsi_var.get():
            columns.append('RSI_14')
        if self.macd_var.get():
            columns.extend(['MACD', 'MACD_Signal', 'MACD_Hist'])
        if self.bb_var.get():
            columns.extend(['BB_Superior', 'BB_Medio', 'BB_Inferior'])
        
        # Reconfigurar o Treeview
        self.data_tree['columns'] = columns
        
        # Mapear nomes amigáveis para as colunas básicas
        column_names = {
            'timestamp': 'Data/Hora',
            'abertura': 'Abertura',
            'alta': 'Alta',
            'baixa': 'Baixa',
            'fechamento': 'Fechamento',
            'volume': 'Volume'
        }
        
        # Reconfigurar os cabeçalhos para TODAS as colunas
        for col in columns:
            if col in column_names:
                # Para as colunas básicas, usar nomes amigáveis
                self.data_tree.heading(col, text=column_names[col])
                if col == 'timestamp':
                    self.data_tree.column(col, width=150)
                else:
                    self.data_tree.column(col, width=80)
            else:
                # Para as colunas de indicadores, usar o próprio nome
                self.data_tree.heading(col, text=col)
                self.data_tree.column(col, width=80)

    def _salvar_dados(self):
        if self.collected_data is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if file_path:
                self.collected_data.to_csv(file_path, index=False)
                self.status_var.set(f"Dados salvos em: {file_path}")
        else:
            messagebox.showwarning("Salvar", "Nenhum dado coletado para salvar.")

    def _limpar_campos(self):
        self.asset_var.set("")
        self.collected_data = None
        self.save_button.config(state=tk.DISABLED)
        self.status_var.set("Pronto para nova coleta")
        
        # Limpar a tabela de dados
        for i in self.data_tree.get_children():
            self.data_tree.delete(i)
