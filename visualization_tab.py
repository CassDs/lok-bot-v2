import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplfinance as mpf

class DataVisualizationTab:
    def __init__(self, master):
        self.frame = ttk.Frame(master)
        self.data = None
        self.data_collection_tab = None  # Referência para a aba de coleta
        self.data_processing_tab = None  # Referência para a aba de processamento
        self._build_ui()

    def set_data_tabs(self, collection_tab, processing_tab):
        """Define referências para as outras abas com dados."""
        self.data_collection_tab = collection_tab
        self.data_processing_tab = processing_tab

    def _build_ui(self):
        """Constrói a interface gráfica da aba de visualização."""
        # Frame para seleção da fonte de dados
        source_frame = ttk.LabelFrame(self.frame, text="Fonte de Dados")
        source_frame.pack(fill="x", padx=10, pady=10)

        # Opções para fonte de dados (botões de rádio)
        self.data_source = tk.StringVar(value="file")
        ttk.Radiobutton(source_frame, text="Carregar de arquivo", 
                      variable=self.data_source, value="file", 
                      command=self._update_buttons).pack(anchor="w", padx=10, pady=2)
        
        ttk.Radiobutton(source_frame, text="Usar dados da coleta atual", 
                      variable=self.data_source, value="collection", 
                      command=self._update_buttons).pack(anchor="w", padx=10, pady=2)
        
        ttk.Radiobutton(source_frame, text="Usar dados processados", 
                      variable=self.data_source, value="processed", 
                      command=self._update_buttons).pack(anchor="w", padx=10, pady=2)

        # Frame para controles de visualização
        control_frame = ttk.LabelFrame(self.frame, text="Visualização de Dados")
        control_frame.pack(fill="x", padx=10, pady=10)

        # Frame para botões de carregamento
        load_frame = ttk.Frame(control_frame)
        load_frame.pack(fill="x", padx=5, pady=5)

        # Botões para carregar dados de diferentes fontes
        self.load_csv_btn = ttk.Button(load_frame, text="Carregar CSV", 
                                     command=self._carregar_csv)
        self.load_csv_btn.pack(side=tk.LEFT, padx=5)
        
        self.load_collection_btn = ttk.Button(load_frame, text="Carregar da Coleta", 
                                           command=self._carregar_da_coleta, state=tk.DISABLED)
        self.load_collection_btn.pack(side=tk.LEFT, padx=5)
        
        self.load_processed_btn = ttk.Button(load_frame, text="Carregar Processados", 
                                          command=self._carregar_processados, state=tk.DISABLED)
        self.load_processed_btn.pack(side=tk.LEFT, padx=5)

        # Frame para configurações do gráfico
        graph_frame = ttk.Frame(control_frame)
        graph_frame.pack(fill="x", padx=5, pady=5)

        # Seleção do tipo de gráfico
        ttk.Label(graph_frame, text="Tipo de Gráfico:").pack(side=tk.LEFT, padx=5)
        self.graph_type_var = tk.StringVar(value="linha")
        ttk.Combobox(graph_frame, textvariable=self.graph_type_var, 
                   values=["linha", "candlestick"], state="readonly").pack(side=tk.LEFT, padx=5)

        # Seleção da coluna para gráfico de linha
        ttk.Label(graph_frame, text="Coluna (linha):").pack(side=tk.LEFT, padx=5)
        self.column_var = tk.StringVar()
        self.column_combobox = ttk.Combobox(graph_frame, textvariable=self.column_var, state="readonly")
        self.column_combobox.pack(side=tk.LEFT, padx=5)

        # Seleção de indicadores técnicos
        ttk.Label(graph_frame, text="Indicadores:").pack(side=tk.LEFT, padx=5)
        self.indicators_var = tk.StringVar()
        self.indicators_combobox = ttk.Combobox(graph_frame, textvariable=self.indicators_var, 
                                              values=["SMA_20", "EMA_9", "RSI", "MACD"], state="readonly")
        self.indicators_combobox.pack(side=tk.LEFT, padx=5)

        # Botão para plotar gráfico
        ttk.Button(graph_frame, text="Plotar Gráfico", command=self._plotar_grafico).pack(side=tk.LEFT, padx=5)

        # Status
        self.status_var = tk.StringVar(value="Nenhum dado carregado.")
        ttk.Label(self.frame, textvariable=self.status_var).pack(pady=5)

        # Área do gráfico
        self.canvas_frame = ttk.Frame(self.frame)
        self.canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def _update_buttons(self):
        """Atualiza o estado dos botões com base na fonte de dados selecionada."""
        source = self.data_source.get()
        
        # Reset all buttons first
        self.load_csv_btn.config(state=tk.DISABLED)
        self.load_collection_btn.config(state=tk.DISABLED)
        self.load_processed_btn.config(state=tk.DISABLED)
        
        # Enable only the relevant button
        if source == "file":
            self.load_csv_btn.config(state=tk.NORMAL)
        elif source == "collection":
            self.load_collection_btn.config(state=tk.NORMAL)
        elif source == "processed":
            self.load_processed_btn.config(state=tk.NORMAL)

    def _carregar_csv(self):
        """Carrega um arquivo CSV e atualiza as opções de colunas."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            self.data = pd.read_csv(file_path, parse_dates=['timestamp'])
            self.data.set_index('timestamp', inplace=True)
            self._calcular_indicadores()  # Calcula indicadores técnicos
            self._atualizar_opcoes_colunas()
            self.status_var.set(f"Dados carregados do arquivo: {file_path}")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar CSV: {str(e)}")

    def _carregar_da_coleta(self):
        """Carrega dados da aba de coleta."""
        if not self.data_collection_tab or self.data_collection_tab.collected_data is None:
            messagebox.showwarning("Aviso", "Nenhum dado coletado disponível.")
            return
            
        try:
            df = self.data_collection_tab.collected_data.copy()
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            self.data = df
            self._calcular_indicadores()  # Calcula indicadores técnicos
            self._atualizar_opcoes_colunas()
            self.status_var.set(f"Dados carregados da coleta: {len(self.data)} registros")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar dados da coleta: {str(e)}")

    def _carregar_processados(self):
        """Carrega dados da aba de processamento."""
        if not self.data_processing_tab or self.data_processing_tab.cleaned_data is None:
            messagebox.showwarning("Aviso", "Nenhum dado processado disponível.")
            return
            
        try:
            df = self.data_processing_tab.cleaned_data.copy()
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            self.data = df
            self._calcular_indicadores()  # Calcula indicadores técnicos
            self._atualizar_opcoes_colunas()
            self.status_var.set(f"Dados processados carregados: {len(self.data)} registros")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar dados processados: {str(e)}")
    
    def _atualizar_opcoes_colunas(self):
        """Atualiza os comboboxes com base nas colunas disponíveis."""
        if self.data is not None:
            # Atualiza as opções de coluna para gráfico de linha
            self.column_combobox['values'] = list(self.data.columns)
            
            # Tenta definir uma coluna padrão sensata
            if 'close' in self.data.columns:
                self.column_var.set('close')
            elif 'fechamento' in self.data.columns:
                self.column_var.set('fechamento')
            elif len(self.data.columns) > 0:
                self.column_var.set(self.data.columns[0])
            
            # Atualiza as opções de indicadores
            indicators = [col for col in self.data.columns if any(
                ind in col for ind in ['SMA', 'EMA', 'RSI', 'MACD', 'BB_'])]
            if indicators:
                self.indicators_combobox['values'] = indicators

    def _calcular_indicadores(self):
        """Calcula indicadores técnicos com base no coleta.py."""
        if self.data is None or 'close' not in self.data.columns:
            return

        df = self.data
        # SMA (Média Móvel Simples)
        df['SMA_20'] = df['close'].rolling(window=20).mean()

        # EMA (Média Móvel Exponencial)
        df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()

        # RSI (Índice de Força Relativa)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)  # Evita divisão por zero
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD (Convergência e Divergência de Médias Móveis)
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        self.data = df.dropna()  # Remove linhas com valores NaN

    def _plotar_grafico(self):
        """Plota o gráfico com base nas seleções do usuário."""
        if self.data is None:
            messagebox.showwarning("Aviso", "Nenhum dado para exibir.")
            return

        # Limpar a área do gráfico
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        tipo_grafico = self.graph_type_var.get()
        
        try:
            if tipo_grafico == "candlestick":
                # Para candlestick, precisamos converter os nomes das colunas
                df_plot = self.data.copy()
                
                # Mapeamento de nomes de colunas
                column_map = {
                    'abertura': 'open',
                    'alta': 'high',
                    'baixa': 'low',
                    'fechamento': 'close',
                    'volume': 'volume'
                }
                
                # Verificar se as colunas originais existem
                required_cols = ['abertura', 'alta', 'baixa', 'fechamento']
                missing_cols = [col for col in required_cols if col not in df_plot.columns]
                
                if missing_cols:
                    # Tentar nomes alternativos (inglês)
                    alternative_map = {
                        'open': 'open',
                        'high': 'high',
                        'low': 'low',
                        'close': 'close'
                    }
                    missing_after_alt = [col for col in ['open', 'high', 'low', 'close'] 
                                        if col not in df_plot.columns]
                    
                    if missing_after_alt:
                        raise ValueError(f"Dados devem conter colunas: abertura, alta, baixa, fechamento "
                                        f"ou open, high, low, close")
                else:
                    # Renomear as colunas
                    df_plot = df_plot.rename(columns=column_map)
                
                # Criar figura para gráfico de candlestick
                # Não use plt.figure() para o mplfinance, ele gerencia sua própria figura
                
                # Verificar se tem volume
                has_volume = 'volume' in df_plot.columns
                
                # Usar mplfinance diretamente, sem criar figura ou axes manualmente
                if has_volume:
                    fig, axes = mpf.plot(
                        df_plot,
                        type='candle',
                        style='charles',
                        title='Gráfico de Candlestick',
                        ylabel='Preço',
                        volume=True,  # Apenas indique que quer volume
                        returnfig=True  # Retorna a figura e os axes para adicionar ao canvas
                    )
                else:
                    fig, axes = mpf.plot(
                        df_plot,
                        type='candle', 
                        style='charles',
                        title='Gráfico de Candlestick',
                        ylabel='Preço',
                        returnfig=True
                    )
                    
            else:  # Gráfico de linha
                # Criar uma figura para o gráfico de linha
                fig = plt.figure(figsize=(10, 6))
                
                coluna = self.column_var.get()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(self.data.index, self.data[coluna], label=coluna)
                ax.set_title(f'Gráfico de Linha: {coluna}')
                ax.set_ylabel('Valor')
                ax.grid(True)
                ax.legend()
                
                # Adicionar indicadores técnicos se selecionados
                indicador = self.indicators_var.get()
                if indicador and indicador in self.data.columns:
                    ax.plot(self.data.index, self.data[indicador], label=indicador)
                    ax.legend()
            
            # Rotacionar as datas no eixo x para melhor visualização
            if tipo_grafico == "linha":
                plt.xticks(rotation=45)
                plt.tight_layout()
            
            # Adicionar o gráfico ao canvas
            canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao plotar gráfico: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Visualização de Dados")
    tab = DataVisualizationTab(root)
    tab.frame.pack(fill="both", expand=True)
    root.mainloop()