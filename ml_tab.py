import tkinter as tk
from tkinter import ttk, filedialog, messagebox, StringVar, BooleanVar, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pickle
import os
import threading
from datetime import datetime
import joblib

# Scikit-learn imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class MachineLearningTab:
    def __init__(self, master, app=None):
        self.frame = ttk.Frame(master)
        self.app = app
        
        # Variáveis de dados e modelo
        self.data = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.predictions = None
        self.scaler = None
        self.selected_features = []
        self.best_params = None

        # Construir a interface
        self._build_ui()

    def _create_scrollable_frame(self, parent):
        """Cria um frame com barra de rolagem vertical"""
        # Container principal
        container = ttk.Frame(parent)
        
        # Canvas para comportar o scrollbar
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        
        # Frame scrollável dentro do canvas
        scrollable_frame = ttk.Frame(canvas)
        
        # Configurar o frame para expandir conforme necessário
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        
        # Associar a barra de rolagem ao canvas
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Criar janela no canvas para o frame scrollável
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # Expandir a largura do frame para preencher o canvas
        def _configure_canvas(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind('<Configure>', _configure_canvas)
        
        # Permitir rolagem com a roda do mouse
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Layout do container, canvas e scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Adicionar atributo scrollable_frame ao container para acesso
        container.scrollable_frame = scrollable_frame
        
        return container
        
    def _build_ui(self):
        """Constrói a interface completa da aba com suporte a rolagem"""
        # Notebook para organizar as seções
        self.notebook = ttk.Notebook(self.frame)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Criar as abas dentro da aba principal - usando frames com scroll
        self.data_tab = self._create_scrollable_frame(self.notebook)
        self.model_tab = self._create_scrollable_frame(self.notebook)
        self.results_tab = self._create_scrollable_frame(self.notebook)
        self.prediction_tab = self._create_scrollable_frame(self.notebook)
        
        # Adicionar as abas ao notebook
        self.notebook.add(self.data_tab, text="Dados")
        self.notebook.add(self.model_tab, text="Treinamento")
        self.notebook.add(self.results_tab, text="Resultados")
        self.notebook.add(self.prediction_tab, text="Previsão")
        
        # Construir o conteúdo de cada aba - passar o frame interno
        self._build_data_tab(self.data_tab.scrollable_frame)
        self._build_model_tab(self.model_tab.scrollable_frame)
        self._build_results_tab(self.results_tab.scrollable_frame)
        self._build_prediction_tab(self.prediction_tab.scrollable_frame)
        
        # Barra de status
        self.status_var = StringVar(value="Pronto")
        status_bar = ttk.Label(self.frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _build_data_tab(self, parent):
        """Constrói a aba de carregamento e preparação de dados"""
        # Frame para origem dos dados
        data_source_frame = ttk.LabelFrame(parent, text="Origem dos Dados")
        data_source_frame.pack(fill="x", padx=10, pady=10)

        # Botões para carregar dados
        button_frame = ttk.Frame(data_source_frame)
        button_frame.pack(padx=10, pady=10)
        
        ttk.Button(button_frame, text="Carregar CSV", 
                command=self._carregar_csv).pack(side=tk.LEFT, padx=5)
                
        ttk.Button(button_frame, text="Usar Dados Coletados", 
                command=self._usar_dados_coletados).pack(side=tk.LEFT, padx=5)
                
        ttk.Button(button_frame, text="Usar Dados Processados", 
                command=self._usar_dados_processados).pack(side=tk.LEFT, padx=5)
        
        # Informações sobre os dados - CORRIGIDO: usar parent em vez de self.data_tab
        info_frame = ttk.LabelFrame(parent, text="Informações dos Dados")
        info_frame.pack(fill="x", padx=10, pady=10)
        
        self.data_info_text = tk.Text(info_frame, height=5, width=80)
        self.data_info_text.pack(fill="both", padx=5, pady=5)
        self.data_info_text.insert(tk.END, "Nenhum dado carregado")
        self.data_info_text.config(state=tk.DISABLED)
        
        # Corrigir as chamadas de métodos abaixo para usar o parent
        self._build_technical_indicators(parent)
        self._build_feature_selection(parent)
    
    def _build_technical_indicators(self, parent):
        """Adiciona seção para criação de indicadores técnicos"""
        # CORRIGIDO: usar parent em vez de self.data_tab
        indicators_frame = ttk.LabelFrame(parent, text="Adicionar Indicadores Técnicos")
        indicators_frame.pack(fill="x", padx=10, pady=10)
        
        # Explicação amigável
        ttk.Label(indicators_frame, 
                text="Os indicadores técnicos podem melhorar a precisão do modelo",
                font=("Helvetica", 9, "italic")).pack(anchor="w", padx=5, pady=5)
        
        # Primeira linha - tipo e período
        row1 = ttk.Frame(indicators_frame)
        row1.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(row1, text="Tipo:").pack(side=tk.LEFT, padx=5)
        
        self.indicator_type_var = StringVar(value="SMA")
        
        # Integrar todos os indicadores em um único combobox
        indicator_types = [
            # Indicadores básicos
            "SMA - Média Móvel Simples",
            "EMA - Média Móvel Exponencial", 
            "RSI - Índice de Força Relativa", 
            "MACD - Convergência/Divergência de Médias Móveis", 
            "Bollinger Bands - Bandas de Bollinger",
            # Indicadores avançados
            "ATR - Amplitude Média Verdadeira",
            "Parkinson - Volatilidade de Parkinson", 
            "Adaptive_RSI - RSI Adaptativo",
            "Force_Index - Índice de Força"
        ]
        
        ttk.Combobox(row1, textvariable=self.indicator_type_var, 
                    values=indicator_types, width=40, state="readonly").pack(side=tk.LEFT, padx=5)
        
        ttk.Label(row1, text="Período:").pack(side=tk.LEFT, padx=(20,5))
        self.indicator_period_var = StringVar(value="14")
        ttk.Entry(row1, textvariable=self.indicator_period_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # Botões para adicionar indicadores
        row2 = ttk.Frame(indicators_frame)
        row2.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(row2, text="Adicionar Indicador", 
                command=self._add_technical_indicator).pack(side=tk.LEFT, padx=5)
        
        # Lista de indicadores adicionados
        row3 = ttk.Frame(indicators_frame)
        row3.pack(fill="x", padx=5, pady=10)
        
        ttk.Label(row3, text="Indicadores adicionados:").pack(anchor="w", padx=5, pady=5)
        
        # Frame para lista com scrollbar
        list_frame = ttk.Frame(row3)
        list_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.indicators_list = tk.Listbox(list_frame, height=5)
        self.indicators_list.pack(side=tk.LEFT, fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.indicators_list.yview)
        scrollbar.pack(side=tk.RIGHT, fill="y")
        self.indicators_list.config(yscrollcommand=scrollbar.set)
        
        ttk.Button(row3, text="Remover Selecionado", 
                command=self._remove_technical_indicator).pack(anchor="w", padx=5, pady=5)
        
    def _add_selected_advanced_indicator(self):
        """Adiciona o indicador avançado selecionado no ComboBox"""
        selected = self.advanced_indicator_var.get()
        
        # Extrair o código do indicador (parte antes do primeiro espaço)
        indicator_code = selected.split(' - ')[0]
        
        if indicator_code == "Time_Features":
            # Chamar o método de features de tempo diretamente
            self._add_time_features()
        else:
            # Chamar o método para outros indicadores avançados
            self._add_advanced_indicator(indicator_code)

    def _add_advanced_indicator(self, indicator_type):
        """Adiciona indicadores técnicos avançados aos dados"""
        if self.data is None:
            messagebox.showwarning("Aviso", "Carregue dados primeiro")
            return
            
        # Encontrar coluna de preço para aplicar o indicador
        price_col = self._get_price_column()
        if not price_col:
            messagebox.showwarning("Aviso", "Não foi possível identificar uma coluna de preço")
            return
        
        try:
            period = int(self.indicator_period_var.get())
            if period <= 0:
                messagebox.showwarning("Aviso", "O período deve ser maior que zero")
                return
        except ValueError:
            messagebox.showwarning("Aviso", "O período deve ser um número inteiro")
            return
            
        try:
            # Implementar cada indicador avançado
            if indicator_type == "ATR":
                # Average True Range - mede volatilidade
                high_col = 'alta' if 'alta' in self.data.columns else price_col
                low_col = 'baixa' if 'baixa' in self.data.columns else price_col
                
                # Cálculo do True Range
                tr1 = self.data[high_col] - self.data[low_col]
                tr2 = abs(self.data[high_col] - self.data[price_col].shift(1))
                tr3 = abs(self.data[low_col] - self.data[price_col].shift(1))
                
                # True Range é o máximo dos três
                tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
                
                # ATR é a média móvel do True Range
                self.data[f'ATR_{period}'] = tr.rolling(window=period).mean()
                indicator_name = f"ATR_{period}"
                
            elif indicator_type == "Parkinson":
                # Volatilidade de Parkinson (usa high/low)
                high_col = 'alta' if 'alta' in self.data.columns else price_col
                low_col = 'baixa' if 'baixa' in self.data.columns else price_col
                
                # Fator de normalização
                norm_factor = 1.0 / (4.0 * np.log(2.0))
                
                # Cálculo da volatilidade
                hl_ratio = (self.data[high_col] / self.data[low_col]).apply(np.log)**2
                self.data[f'Parkinson_{period}'] = np.sqrt(norm_factor * hl_ratio.rolling(window=period).mean())
                indicator_name = f"Parkinson_{period}"
                
            elif indicator_type == "Adaptive_RSI":
                # RSI Adaptativo - ajusta o período com base na volatilidade
                delta = self.data[price_col].diff()
                
                # Calcular volatilidade para adaptação do período
                volatility = delta.abs().rolling(window=period).mean()
                
                # Adaptar período base no nível de volatilidade
                adaptive_period = (period * (1 + volatility / volatility.rolling(window=period*2).mean())).fillna(period).astype(int)
                adaptive_period = adaptive_period.clip(lower=5, upper=period*2)  # limitar mudanças extremas
                
                # Precalcular ganhos e perdas
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                # Aplicar o período adaptativo (aproximação)
                rs_values = []
                for i in range(len(self.data)):
                    if i < period:
                        rs_values.append(np.nan)
                        continue
                    
                    p = adaptive_period.iloc[i]
                    avg_gain = gain.iloc[max(0, i-p):i].mean()
                    avg_loss = loss.iloc[max(0, i-p):i].mean()
                    
                    if avg_loss == 0:
                        rs = 100.0
                    else:
                        rs = avg_gain / avg_loss
                    
                    rs_values.append(rs)
                    
                self.data[f'Adaptive_RSI_{period}'] = 100 - (100 / (1 + pd.Series(rs_values)))
                indicator_name = f"Adaptive_RSI_{period}"
                
            elif indicator_type == "Force_Index":
                # Índice de Força - combina preço e volume
                volume_col = None
                for col_name in self.data.columns:
                    if 'volume' in col_name.lower():
                        volume_col = col_name
                        break
                
                if volume_col is None:
                    messagebox.showwarning("Aviso", "Dados de volume não encontrados, usando valor constante")
                    self.data['volume'] = 1
                    volume_col = 'volume'
                
                # Calcular Índice de Força
                price_change = self.data[price_col].diff()
                raw_force = price_change * self.data[volume_col]
                
                # EMA do índice de força
                self.data[f'Force_Index_{period}'] = raw_force.ewm(span=period, adjust=False).mean()
                indicator_name = f"Force_Index_{period}"
                
            else:
                messagebox.showwarning("Aviso", f"Indicador '{indicator_type}' não implementado")
                return
                
            # Adicionar à lista de indicadores
            self.indicators_list.insert(tk.END, f"{indicator_name} ({price_col})")
            
            # Remover linhas com NaN criadas pelos indicadores
            self.data.dropna(inplace=True)
            
            # Atualizar informações e listas
            self._atualizar_info_dados("Dados atualizados com novo indicador avançado")
            self._atualizar_listas_features()
            
            self.status_var.set(f"Indicador avançado {indicator_name} adicionado com sucesso")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao adicionar indicador avançado: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _build_feature_selection(self, parent):
        """Constrói a interface para seleção de características"""
        # CORRIGIDO: usar parent em vez de self.data_tab
        feature_frame = ttk.LabelFrame(parent, text="Selecionar Colunas para Treinamento")
        feature_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
        # Explicação amigável
        ttk.Label(feature_frame, 
                 text="Selecione quais colunas o modelo deve usar para fazer previsões",
                 font=("Helvetica", 9, "italic")).pack(anchor="w", padx=5, pady=5)
        
        # Colunas disponíveis e selecionadas
        list_frame = ttk.Frame(feature_frame)
        list_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Lado esquerdo - Colunas disponíveis
        left_frame = ttk.LabelFrame(list_frame, text="Colunas Disponíveis")
        left_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=5, pady=5)
        
        self.available_columns = tk.Listbox(left_frame, height=10, selectmode=tk.EXTENDED)
        self.available_columns.pack(side=tk.LEFT, fill="both", expand=True)
        
        avail_scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=self.available_columns.yview)
        avail_scrollbar.pack(side=tk.RIGHT, fill="y")
        self.available_columns.config(yscrollcommand=avail_scrollbar.set)
        
        # Botões do meio
        mid_frame = ttk.Frame(list_frame)
        mid_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        ttk.Button(mid_frame, text=">>", command=self._add_all_columns).pack(pady=5)
        ttk.Button(mid_frame, text=">", command=self._add_selected_columns).pack(pady=5)
        ttk.Button(mid_frame, text="<", command=self._remove_selected_columns).pack(pady=5)
        ttk.Button(mid_frame, text="<<", command=self._remove_all_columns).pack(pady=5)
        
        # Lado direito - Colunas selecionadas
        right_frame = ttk.LabelFrame(list_frame, text="Colunas Selecionadas")
        right_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=5, pady=5)
        
        self.selected_columns = tk.Listbox(right_frame, height=10, selectmode=tk.EXTENDED)
        self.selected_columns.pack(side=tk.LEFT, fill="both", expand=True)
        
        sel_scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=self.selected_columns.yview)
        sel_scrollbar.pack(side=tk.RIGHT, fill="y")
        self.selected_columns.config(yscrollcommand=sel_scrollbar.set)
    
    def _build_model_tab(self, parent):
        """Constrói a aba de configuração e treinamento do modelo"""
        
        # Frame para configuração do alvo (target)
        target_frame = ttk.LabelFrame(parent, text="O que você quer prever?")
        target_frame.pack(fill="x", padx=10, pady=10)
        
        # Explicação amigável
        ttk.Label(target_frame, 
                text="Escolha o tipo de previsão que o modelo fará",
                font=("Helvetica", 9, "italic")).pack(anchor="w", padx=5, pady=5)
        
        # Tipo de previsão
        type_frame = ttk.Frame(target_frame)
        type_frame.pack(fill="x", padx=5, pady=5)
        
        self.target_type_var = StringVar(value="next_candle")
        ttk.Radiobutton(type_frame, text="Próxima vela (sobe ou desce)", 
                       variable=self.target_type_var, value="next_candle").pack(anchor="w", padx=5, pady=2)
        
        ttk.Radiobutton(type_frame, text="Tendência futura (depois de vários períodos)", 
                       variable=self.target_type_var, value="trend").pack(anchor="w", padx=5, pady=2)
        
        # Período para tendência
        trend_frame = ttk.Frame(target_frame)
        trend_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(trend_frame, text="Quantas velas no futuro (para tendência):").pack(side=tk.LEFT, padx=5)
        self.trend_periods_var = StringVar(value="5")
        ttk.Entry(trend_frame, textvariable=self.trend_periods_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # NOVO: Timeframe da previsão
        timeframe_frame = ttk.Frame(target_frame)
        timeframe_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(timeframe_frame, text="Período da vela:").pack(side=tk.LEFT, padx=5)
        self.timeframe_var = StringVar(value="1m")
        timeframe_options = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]
        ttk.Combobox(timeframe_frame, textvariable=self.timeframe_var, 
                    values=timeframe_options, width=5, state="readonly").pack(side=tk.LEFT, padx=5)
        
        # Explicação adicional
        ttk.Label(timeframe_frame, 
                text="Selecione o timeframe que está sendo previsto (ex: 5m = velas de 5 minutos)",
                font=("Helvetica", 9, "italic")).pack(side=tk.LEFT, padx=10)
        
        # Frame para divisão dos dados
        split_frame = ttk.LabelFrame(parent, text="Divisão dos Dados")
        split_frame.pack(fill="x", padx=10, pady=10)
        
        # Explicação amigável
        ttk.Label(split_frame, 
                text="Divisão dos dados entre treino (para aprendizado) e teste (para verificação)",
                font=("Helvetica", 9, "italic")).pack(anchor="w", padx=5, pady=5)
        
        # Porcentagem de treino
        split_row = ttk.Frame(split_frame)
        split_row.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(split_row, text="Porcentagem para treino:").pack(side=tk.LEFT, padx=5)
        self.train_size_var = StringVar(value="80")
        ttk.Entry(split_row, textvariable=self.train_size_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(split_row, text="%").pack(side=tk.LEFT)
        
        # Opção temporal vs aleatória
        split_method_frame = ttk.Frame(split_frame)
        split_method_frame.pack(fill="x", padx=5, pady=5)
        
        self.split_method_var = StringVar(value="time")
        ttk.Radiobutton(split_method_frame, text="Divisão temporal simples", 
                      variable=self.split_method_var, value="time").pack(anchor="w", padx=5, pady=2)
        
        ttk.Radiobutton(split_method_frame, text="Divisão aleatória", 
                      variable=self.split_method_var, value="random").pack(anchor="w", padx=5, pady=2)
                      
        ttk.Radiobutton(split_method_frame, text="Validação cruzada temporal (recomendado)", 
                      variable=self.split_method_var, value="time_series_cv").pack(anchor="w", padx=5, pady=2)
        
        # Configurações de validação cruzada
        cv_frame = ttk.Frame(split_method_frame)
        cv_frame.pack(fill="x", padx=25, pady=5)
        
        ttk.Label(cv_frame, text="Número de folds:").pack(side=tk.LEFT, padx=5)
        self.n_splits_var = StringVar(value="5")
        ttk.Entry(cv_frame, textvariable=self.n_splits_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # Normalizar dados
        norm_frame = ttk.Frame(split_frame)
        norm_frame.pack(fill="x", padx=5, pady=5)
        
        self.normalize_var = BooleanVar(value=True)
        ttk.Checkbutton(norm_frame, text="Normalizar dados (recomendado)", 
                      variable=self.normalize_var).pack(anchor="w", padx=5)
        
        # Frame para seleção do modelo
        model_frame = ttk.LabelFrame(parent, text="Escolha do Modelo")
        model_frame.pack(fill="x", padx=10, pady=10)
        
        # Explicação amigável
        ttk.Label(model_frame, 
                text="Escolha o algoritmo de aprendizado de máquina",
                font=("Helvetica", 9, "italic")).pack(anchor="w", padx=5, pady=5)
        
        # Tipo de modelo
        model_row = ttk.Frame(model_frame)
        model_row.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(model_row, text="Algoritmo:").pack(side=tk.LEFT, padx=5)
        self.algorithm_var = StringVar(value="RandomForest")
        
        algorithms = [
            "LogisticRegression - Simples e rápido",
            "RandomForest - Equilibrado e preciso",
            "GradientBoosting - Alta precisão, mais lento",
            "XGBoost - Alta performance"
        ]
        
        ttk.Combobox(model_row, textvariable=self.algorithm_var, 
                   values=algorithms, width=40, state="readonly").pack(side=tk.LEFT, padx=5)
        
        # Adicionar seção de modelo existente
        existing_model_frame = ttk.LabelFrame(parent, text="Modelo Existente")
        existing_model_frame.pack(fill="x", padx=10, pady=10)
        
        # Explicação amigável
        ttk.Label(existing_model_frame, 
                text="Carregue um modelo existente ou continue treinando um modelo atual",
                font=("Helvetica", 9, "italic")).pack(anchor="w", padx=5, pady=5)
        
        buttons_frame = ttk.Frame(existing_model_frame)
        buttons_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(buttons_frame, text="Salvar Modelo", 
                  command=self._salvar_modelo).pack(side=tk.LEFT, padx=5)
                  
        ttk.Button(buttons_frame, text="Carregar Modelo", 
                  command=self._carregar_modelo).pack(side=tk.LEFT, padx=5)
                  
        ttk.Button(buttons_frame, text="Continuar Treinamento", 
                  command=self._continuar_treinamento).pack(side=tk.LEFT, padx=5)
        
        # Status do modelo atual
        self.model_info_var = StringVar(value="Nenhum modelo carregado")
        ttk.Label(existing_model_frame, textvariable=self.model_info_var).pack(anchor="w", padx=5, pady=5)
        
        # Adicionar seção para otimização de hiperparâmetros
        tuning_frame = ttk.LabelFrame(parent, text="Otimização de Hiperparâmetros")
        tuning_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(tuning_frame, text="A otimização automática pode melhorar significativamente o desempenho do modelo",
                font=("Helvetica", 9, "italic")).pack(anchor="w", padx=5, pady=5)
        
        self.tuning_var = BooleanVar(value=False)
        ttk.Checkbutton(tuning_frame, text="Ativar otimização automática de hiperparâmetros", 
                     variable=self.tuning_var).pack(anchor="w", padx=5, pady=5)
        
        tuning_options = ttk.Frame(tuning_frame)
        tuning_options.pack(fill="x", padx=20, pady=5)
        
        ttk.Label(tuning_options, text="Método:").pack(side=tk.LEFT, padx=5)
        self.tuning_method_var = StringVar(value="random")
        ttk.Combobox(tuning_options, textvariable=self.tuning_method_var, 
                   values=["random", "grid"], width=10, state="readonly").pack(side=tk.LEFT, padx=5)
        
        ttk.Label(tuning_options, text="Iterações:").pack(side=tk.LEFT, padx=(15,5))
        self.tuning_iter_var = StringVar(value="10")
        ttk.Entry(tuning_options, textvariable=self.tuning_iter_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(tuning_options, text="CV:").pack(side=tk.LEFT, padx=(15,5))
        self.tuning_cv_var = StringVar(value="5")
        ttk.Entry(tuning_options, textvariable=self.tuning_cv_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # Botão de treinamento
        train_button_frame = ttk.Frame(parent)
        train_button_frame.pack(pady=20)
        
        self.train_button = ttk.Button(train_button_frame, text="Treinar Modelo", 
                                     command=self._treinar_modelo, 
                                     style="Accent.TButton" if hasattr(ttk, "Accent.TButton") else "")
        self.train_button.pack(pady=10, ipadx=20, ipady=5)
        
        # Barra de progresso
        self.progress = ttk.Progressbar(parent, orient="horizontal", mode="indeterminate")
        self.progress.pack(fill="x", padx=10, pady=5)

    def _add_time_features(self):
        """Adiciona features de tempo aos dados (hora do dia, dia da semana, etc.)"""
        if self.data is None:
            messagebox.showwarning("Aviso", "Carregue dados primeiro")
            return
            
        # Verificar se existe coluna de timestamp
        if 'timestamp' not in self.data.columns:
            messagebox.showwarning("Aviso", "Os dados não possuem coluna de timestamp")
            return
            
        try:
            # Garantir que a coluna timestamp é do tipo datetime
            if not pd.api.types.is_datetime64_any_dtype(self.data['timestamp']):
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                
            # Criar features de tempo
            # Hora do dia (0-23)
            self.data['hour'] = self.data['timestamp'].dt.hour
            
            # Minuto da hora (0-59)
            self.data['minute'] = self.data['timestamp'].dt.minute
            
            # Dia da semana (0=segunda, 6=domingo)
            self.data['day_of_week'] = self.data['timestamp'].dt.dayofweek
            
            # Features cíclicas para hora (representação circular do tempo)
            self.data['hour_sin'] = np.sin(2 * np.pi * self.data['hour'] / 24)
            self.data['hour_cos'] = np.cos(2 * np.pi * self.data['hour'] / 24)
            
            # Período do dia (categorias)
            conditions = [
                (self.data['hour'] >= 0) & (self.data['hour'] < 6),
                (self.data['hour'] >= 6) & (self.data['hour'] < 12),
                (self.data['hour'] >= 12) & (self.data['hour'] < 18),
                (self.data['hour'] >= 18)
            ]
            periods = ['madrugada', 'manhã', 'tarde', 'noite']
            self.data['day_period'] = np.select(conditions, periods)
            
            # Features para sessões de mercado (mercado forex)
            # Sessão de Tóquio: 00:00-09:00 UTC
            self.data['tokyo_session'] = ((self.data['hour'] >= 0) & (self.data['hour'] < 9)).astype(int)
            # Sessão de Londres: 08:00-17:00 UTC
            self.data['london_session'] = ((self.data['hour'] >= 8) & (self.data['hour'] < 17)).astype(int)
            # Sessão de Nova York: 13:00-22:00 UTC
            self.data['ny_session'] = ((self.data['hour'] >= 13) & (self.data['hour'] < 22)).astype(int)
            # Sobreposição Londres-NY: 13:00-17:00 UTC (alta liquidez)
            self.data['overlap_session'] = ((self.data['hour'] >= 13) & (self.data['hour'] < 17)).astype(int)
            
            # Remover features categóricas que não são úteis diretamente para ML
            self.data.drop(['day_period'], axis=1, inplace=True)
            
            # Converter dia da semana para variáveis dummy
            for day in range(7):
                self.data[f'weekday_{day}'] = (self.data['day_of_week'] == day).astype(int)
            
            # Remover colunas originais
            self.data.drop(['hour', 'minute', 'day_of_week'], axis=1, inplace=True)
            
            # Atualizar informações e listas
            self._atualizar_info_dados("Dados atualizados com features de tempo")
            self._atualizar_listas_features()
            
            # Adicionar à lista de indicadores
            self.indicators_list.insert(tk.END, "Features de Tempo")
            
            self.status_var.set("Features de tempo adicionadas com sucesso")
            messagebox.showinfo("Sucesso", "Features de tempo adicionadas:\n"
                            "• Representações cíclicas da hora\n"
                            "• Dias da semana\n"
                            "• Sessões de mercado (Tóquio, Londres, NY)\n"
                            "• Sobreposição de sessões")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao adicionar features de tempo: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _build_advanced_training_options(self, parent):
        """Adiciona opções avançadas de treinamento"""
        advanced_frame = ttk.LabelFrame(parent, text="Opções Avançadas de Treinamento")
        advanced_frame.pack(fill="x")
        
        ttk.Label(advanced_frame, text="Estratégias específicas para dados financeiros",
                font=("Helvetica", 9, "italic")).pack(anchor="w", padx=5, pady=5)
    
        
        # Opções de regime de mercado
        regime_frame = ttk.Frame(advanced_frame)
        regime_frame.pack(fill="x", padx=5, pady=5)
        
        self.market_regime_var = BooleanVar(value=False)
        ttk.Checkbutton(regime_frame, text="Detectar e tratar regimes de mercado separadamente", 
                    variable=self.market_regime_var).pack(anchor="w", padx=5)
        
        # Opções de janela deslizante
        window_frame = ttk.Frame(advanced_frame)
        window_frame.pack(fill="x", padx=5, pady=5)
        
        self.sliding_window_var = BooleanVar(value=False)
        ttk.Checkbutton(window_frame, text="Usar janela deslizante para priorizar dados recentes", 
                    variable=self.sliding_window_var).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(window_frame, text="Tamanho da janela:").pack(side=tk.LEFT, padx=(15,5))
        self.window_size_var = StringVar(value="100")
        ttk.Entry(window_frame, textvariable=self.window_size_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # Opções de balanceamento de classes
        balance_frame = ttk.Frame(advanced_frame)
        balance_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(balance_frame, text="Balanceamento de classes:").pack(side=tk.LEFT, padx=5)
        self.balance_method_var = StringVar(value="none")
        ttk.Combobox(balance_frame, textvariable=self.balance_method_var, 
                    values=["none", "undersample", "oversample", "SMOTE"], 
                    width=10, state="readonly").pack(side=tk.LEFT, padx=5)
    
    def _plot_learning_curve(self):
        """Plota curva de aprendizado para avaliar overfitting"""
        if not hasattr(self, 'model') or self.model is None:
            messagebox.showwarning("Aviso", "Treine um modelo primeiro")
            return
            
        # Limpar frame do gráfico
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        # Mostrar uma mensagem de progresso
        status_label = ttk.Label(self.plot_frame, text="Calculando curva de aprendizado...\nIsso pode levar alguns segundos.", 
                            font=("Helvetica", 10))
        status_label.pack(pady=20)
        self.plot_frame.update()
        
        try:
            from sklearn.model_selection import learning_curve
            
            # Criar conjunto de treinos de diferentes tamanhos
            train_sizes, train_scores, test_scores = learning_curve(
                self.model, self.X_train, self.y_train, cv=5, 
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='accuracy'
            )
            
            # Calcular médias e desvios
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            # Remover o label de status
            status_label.destroy()
            
            # Criar figura
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plotar curva de aprendizado
            ax.set_title("Curva de Aprendizado", fontsize=14)
            ax.set_xlabel("Tamanho do Conjunto de Treino", fontsize=12)
            ax.set_ylabel("Acurácia", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
            ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
            ax.plot(train_sizes, train_mean, 'o-', color="r", label="Treino", linewidth=2)
            ax.plot(train_sizes, test_mean, 'o-', color="g", label="Validação", linewidth=2)
            
            # Adicionar linha para gap de overfitting
            gap = train_mean - test_mean
            ax.plot(train_sizes, gap, '--', color="b", label="Gap (overfitting)", linewidth=1.5)
            
            # Destacar overfitting se o gap for muito grande
            max_gap = np.max(gap)
            if max_gap > 0.1:
                overfitting_idx = np.argmax(gap)
                ax.annotate(f'Overfitting\nGap: {max_gap:.3f}',
                        xy=(train_sizes[overfitting_idx], gap[overfitting_idx]),
                        xytext=(train_sizes[overfitting_idx], gap[overfitting_idx] + 0.1),
                        arrowprops=dict(facecolor='black', shrink=0.05),
                        fontsize=10, ha='center')
            
            ax.legend(loc="best", fontsize=10)
            
            # Mostrar na interface
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.pack(fill=tk.BOTH, expand=True)
            
            # Adicionar barra de navegação
            toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
            toolbar.update()
            
            # Adicionar interpretação
            interpretation_frame = ttk.Frame(self.plot_frame)
            interpretation_frame.pack(fill="x", pady=10)
            
            interpretation_text = (
                "Interpretação da Curva de Aprendizado:\n"
                "• Se o gap entre treino e teste é grande → Overfitting (modelo muito complexo)\n"
                "• Se ambas as curvas estão baixas → Underfitting (modelo muito simples)\n"
                "• Se as curvas convergem com alta acurácia → Modelo bem balanceado"
            )
            
            ttk.Label(interpretation_frame, text=interpretation_text, 
                    background="lightyellow", padding=10, 
                    wraplength=600, justify="left").pack(fill="x")
        
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar curva de aprendizado: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _build_results_tab(self, parent):
        """Constrói a aba de visualização de resultados"""
        # Frame para métricas de texto
        metrics_frame = ttk.LabelFrame(parent, text="Resultados do Modelo")
        metrics_frame.pack(fill="x", padx=10, pady=10)
        
        # Área de texto com resultados
        self.results_text = scrolledtext.ScrolledText(metrics_frame, height=10, width=80, wrap=tk.WORD)
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.results_text.insert(tk.END, "Treine um modelo para ver os resultados aqui")
        self.results_text.config(state=tk.DISABLED)
        
        # Visualizações
        viz_frame = ttk.LabelFrame(parent, text="Visualizações")
        viz_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Botões para diferentes visualizações
        button_frame = ttk.Frame(viz_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(button_frame, text="Matriz de Confusão", 
                  command=self._plot_confusion_matrix).pack(side=tk.LEFT, padx=5)
                  
        ttk.Button(button_frame, text="Importância das Features", 
                  command=self._plot_feature_importance).pack(side=tk.LEFT, padx=5)
        
        # Adicionar o novo botão para Curva de Aprendizado
        ttk.Button(button_frame, text="Curva de Aprendizado", 
                  command=self._plot_learning_curve).pack(side=tk.LEFT, padx=5)
        
        # Frame para o gráfico
        self.plot_frame = ttk.Frame(viz_frame)
        self.plot_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Validação temporal
        ttk.Button(button_frame, text="Validação Temporal", 
                command=self._plot_temporal_validation).pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Monitorar Overfitting", 
                command=self._monitor_overfitting).pack(side=tk.LEFT, padx=5)

    def _plot_temporal_validation(self):
        """Visualiza resultados de validação temporal"""
        if not hasattr(self, 'model') or self.model is None:
            messagebox.showwarning("Aviso", "Treine um modelo primeiro")
            return
        
        # Limpar frame do gráfico
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        try:
            # Mostrar uma mensagem de progresso
            status_label = ttk.Label(self.plot_frame, 
                                text="Calculando validação temporal...\nIsso pode levar alguns segundos.", 
                                font=("Helvetica", 10))
            status_label.pack(pady=20)
            self.plot_frame.update()
            
            # Executar validação cruzada temporal
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = []
            
            for train_idx, test_idx in tscv.split(self.X_train):
                # Ajustar aos índices dependendo do tipo de dados
                if isinstance(self.X_train, np.ndarray):
                    X_fold_train, X_fold_test = self.X_train[train_idx], self.X_train[test_idx]
                    y_fold_train, y_fold_test = self.y_train[train_idx], self.y_train[test_idx]
                else:
                    X_fold_train, X_fold_test = self.X_train.iloc[train_idx], self.X_train.iloc[test_idx]
                    y_fold_train, y_fold_test = self.y_train.iloc[train_idx], self.y_train.iloc[test_idx]
                
                # Criar uma cópia do modelo para não interferir no modelo principal
                from sklearn.base import clone
                fold_model = clone(self.model)
                
                # Treinar e avaliar
                fold_model.fit(X_fold_train, y_fold_train)
                score = fold_model.score(X_fold_test, y_fold_test)
                cv_scores.append(score)
            
            # Remover label de status
            status_label.destroy()
            
            # Plotar resultados
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Gráfico de barras para cada fold
            bars = ax.bar(range(1, len(cv_scores) + 1), cv_scores, color='skyblue')
            
            # Adicionar linha de média
            mean_score = np.mean(cv_scores)
            ax.axhline(y=mean_score, color='red', linestyle='-', label=f'Média: {mean_score:.4f}')
            
            # Destacar o desvio padrão
            std_score = np.std(cv_scores)
            ax.axhspan(mean_score - std_score, mean_score + std_score, alpha=0.1, color='red',
                    label=f'Desvio Padrão: ±{std_score:.4f}')
            
            # Configurar gráfico
            ax.set_xlabel('Fold de Validação Temporal', fontsize=12)
            ax.set_ylabel('Acurácia', fontsize=12)
            ax.set_title('Validação Cruzada Temporal (Mais recente à direita)', fontsize=14)
            ax.set_ylim([0, 1])
            ax.set_xticks(range(1, len(cv_scores) + 1))
            
            # Adicionar valores sobre as barras
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 pontos de offset vertical
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            
            # CORRIGIDO: Posicionar legenda fora da área do gráfico
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                    fontsize=10, ncol=2, frameon=True, fancybox=True, shadow=True)
                    
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Adicionar explicação - movida para mais abaixo para não conflitar com a legenda
            fig.text(0.5, 0.01, 
                    "Os folds temporais mostram a performance do modelo em diferentes períodos.\n"
                    "Folds mais recentes (à direita) são mais relevantes para performance futura.", 
                    ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.5", 
                    facecolor='lightyellow', alpha=0.5))
            
            # Ajustar layout para acomodar a legenda externa
            plt.subplots_adjust(bottom=0.25)
            
            # Mostrar na interface
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.pack(fill=tk.BOTH, expand=True)
            
            # Barra de navegação
            toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
            toolbar.update()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao calcular validação temporal: {str(e)}")
            import traceback
            traceback.print_exc()

    def _monitor_overfitting(self):
        """Monitora e visualiza overfitting em diferentes partes dos dados"""
        if not hasattr(self, 'model') or self.model is None:
            messagebox.showwarning("Aviso", "Treine um modelo primeiro")
            return
        
        # Limpar frame do gráfico
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        # Preparar os dados
        X = self.X_train if isinstance(self.X_train, np.ndarray) else self.X_train.values
        y = self.y_train if isinstance(self.y_train, np.ndarray) else self.y_train.values
        
        try:
            # Calcular performance em diferentes subconjuntos dos dados
            subsets = 10
            train_scores = []
            test_scores = []
            
            # Criar modelo clone para cada subset
            from sklearn.base import clone
            
            for i in range(1, subsets + 1):
                train_size = int(len(X) * (i / subsets))
                X_subset = X[:train_size]
                y_subset = y[:train_size]
                
                # Treinar modelo no subconjunto
                model_clone = clone(self.model)
                model_clone.fit(X_subset, y_subset)
                
                # Medir performance no treino e teste
                train_score = model_clone.score(X_subset, y_subset)
                test_score = model_clone.score(self.X_test, self.y_test)
                
                train_scores.append(train_score)
                test_scores.append(test_score)
            
            # Criar figura - SOLUÇÃO 1: Aumentar o tamanho vertical da figura
            fig, ax = plt.subplots(figsize=(10, 7))  # Aumentar altura de 6 para 7
            
            # Preparar eixo X - tamanho dos dados
            data_sizes = [(i / subsets) * 100 for i in range(1, subsets + 1)]
            
            # Plotar curvas
            ax.plot(data_sizes, train_scores, 'o-', color='blue', label='Score Treino')
            ax.plot(data_sizes, test_scores, 'o-', color='green', label='Score Teste')
            
            # Plotar gap de overfitting
            overfitting_gap = [train - test for train, test in zip(train_scores, test_scores)]
            ax.plot(data_sizes, overfitting_gap, '--', color='red', label='Gap de Overfitting')
            
            # Configurar gráfico
            ax.set_xlabel('Porcentagem dos Dados Utilizados', fontsize=12)
            ax.set_ylabel('Acurácia', fontsize=12)
            ax.set_title('Monitoramento de Overfitting', fontsize=14)
            ax.set_ylim([0, 1])
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # SOLUÇÃO 2: Mover legenda para o lado direito do gráfico
            ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), 
                    fontsize=10, frameon=True, fancybox=True, shadow=True)
            
            # Marcar tendências preocupantes
            if overfitting_gap[-1] > 0.15:
                ax.text(data_sizes[-1], overfitting_gap[-1], 
                    'ALERTA: Gap >15%',
                    color='red', fontsize=12, ha='right', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
            
            # Mostrar interpretação - movida para mais abaixo
            interpretation = (
                "Interpretação do Monitoramento de Overfitting:\n"
                "• Gap crescente entre treino e teste indica possível overfitting\n"
                "• Curva de teste estável ou ascendente indica boa generalização\n"
                "• Gap > 15% pode indicar que o modelo está decorando os dados"
            )
            
            # SOLUÇÃO 3: Posicionar texto de interpretação mais abaixo
            fig.text(0.5, 0.01, interpretation, ha='center', fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.5))
            
            # SOLUÇÃO 4: Ajustar layout para acomodar a legenda à direita
            plt.tight_layout()  # Primeiro ajustar layout interno
            plt.subplots_adjust(right=0.8, bottom=0.2)  # Depois refinar margens específicas
            
            # Mostrar na interface
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.pack(fill=tk.BOTH, expand=True)
            
            # Barra de navegação
            toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
            toolbar.update()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao monitorar overfitting: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _build_prediction_tab(self, parent):
        """Constrói a aba de previsão com o modelo treinado"""
        # Frame para informações do modelo
        model_frame = ttk.LabelFrame(parent, text="Modelo Atual")
        model_frame.pack(fill="x", padx=10, pady=10)
        
        # Status do modelo
        self.model_status_var = StringVar(value="Nenhum modelo carregado")
        ttk.Label(model_frame, textvariable=self.model_status_var, 
                font=("Helvetica", 10, "bold")).pack(padx=5, pady=5)
        
        # Acurácia do modelo
        self.model_accuracy_var = StringVar(value="Acurácia: N/A")
        ttk.Label(model_frame, textvariable=self.model_accuracy_var).pack(padx=5, pady=5)
        
        # Frame para opções de previsão
        options_frame = ttk.LabelFrame(parent, text="Opções de Previsão")
        options_frame.pack(fill="x", padx=10, pady=10)
        
        # Fonte de dados
        source_frame = ttk.Frame(options_frame)
        source_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(source_frame, text="Fonte de dados:").pack(side=tk.LEFT, padx=5)
        self.prediction_source_var = StringVar(value="recent")
        
        ttk.Radiobutton(source_frame, text="Dados recentes (coleta atual)", 
                    variable=self.prediction_source_var, value="recent").pack(side=tk.LEFT, padx=5)
        
        ttk.Radiobutton(source_frame, text="Dados de teste (mesma base de treino)", 
                    variable=self.prediction_source_var, value="test").pack(side=tk.LEFT, padx=5)
        
        # Número de candles para previsão
        candles_frame = ttk.Frame(options_frame)
        candles_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(candles_frame, text="Número de candles:").pack(side=tk.LEFT, padx=5)
        self.prediction_candles_var = StringVar(value="10")
        ttk.Entry(candles_frame, textvariable=self.prediction_candles_var, width=5).pack(side=tk.LEFT, padx=5)

        # REMOVIDO: Configurações de threshold para operações
        
        # Botão para fazer previsão
        ttk.Button(options_frame, text="Fazer Previsão", 
                command=self._fazer_previsao).pack(pady=10)
        
        # Frame para resultados da previsão
        results_frame = ttk.LabelFrame(parent, text="Resultados da Previsão")
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Texto de resultados
        self.prediction_text = scrolledtext.ScrolledText(results_frame, height=5, width=80, wrap=tk.WORD)
        self.prediction_text.pack(fill="x", padx=5, pady=5)
        self.prediction_text.insert(tk.END, "Faça uma previsão para ver os resultados aqui")
        self.prediction_text.config(state=tk.DISABLED)
        
        # Frame para o gráfico de previsão
        self.prediction_plot_frame = ttk.Frame(results_frame)
        self.prediction_plot_frame.pack(fill="both", expand=True, padx=5, pady=5)
    
    def _carregar_csv(self):
        """Carrega dados de um arquivo CSV"""
        file_path = filedialog.askopenfilename(
            title="Selecione o arquivo CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            self.data = pd.read_csv(file_path)
            
            # Verificar se há coluna de timestamp e converter para datetime
            if 'timestamp' in self.data.columns:
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            
            self._atualizar_info_dados(f"Dados carregados de {file_path}")
            self._atualizar_listas_features()
            self.status_var.set("Dados carregados com sucesso")
            
            # Trocar para a próxima aba
            self.notebook.select(1)  # Índice 1 é a aba "Treinamento"
            
        except Exception as e:
            messagebox.showerror("Erro ao carregar arquivo", f"Ocorreu um erro: {str(e)}")
            self.status_var.set(f"Erro ao carregar arquivo: {str(e)}")
    
    def _usar_dados_coletados(self):
        """Usa os dados da aba de coleta"""
        if not self.app or not hasattr(self.app, 'data_collection_tab'):
            messagebox.showwarning("Aviso", "Aba de coleta de dados não disponível")
            return
            
        if not hasattr(self.app.data_collection_tab, 'collected_data'):
            messagebox.showwarning("Aviso", "Nenhum dado disponível na aba de coleta")
            return
            
        try:
            self.data = self.app.data_collection_tab.collected_data.copy()
            self._atualizar_info_dados("Dados carregados da aba de coleta")
            self._atualizar_listas_features()
            self.status_var.set("Dados da coleta carregados com sucesso")
            
            # Trocar para a próxima aba
            self.notebook.select(1)  # Índice 1 é a aba "Treinamento"
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar dados da coleta: {str(e)}")
    
    def _usar_dados_processados(self):
        """Usa os dados processados da aba de processamento"""
        if not self.app or not hasattr(self.app, 'processing_tab'):
            messagebox.showwarning("Aviso", "Aba de processamento não disponível")
            return
            
        if not hasattr(self.app.processing_tab, 'cleaned_data'):
            messagebox.showwarning("Aviso", "Nenhum dado processado disponível")
            return
            
        try:
            self.data = self.app.processing_tab.cleaned_data.copy()
            self._atualizar_info_dados("Dados carregados da aba de processamento")
            self._atualizar_listas_features()
            self.status_var.set("Dados processados carregados com sucesso")
            
            # Trocar para a próxima aba
            self.notebook.select(1)  # Índice 1 é a aba "Treinamento"
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar dados processados: {str(e)}")
    
    def _atualizar_info_dados(self, mensagem):
        """Atualiza a caixa de informações sobre os dados"""
        self.data_info_text.config(state=tk.NORMAL)
        self.data_info_text.delete('1.0', tk.END)
        
        if self.data is not None:
            info = f"{mensagem}\n\n"
            info += f"Formato dos dados: {self.data.shape[0]} linhas x {self.data.shape[1]} colunas\n"
            
            if 'timestamp' in self.data.columns:
                info += f"Período: {self.data['timestamp'].min()} a {self.data['timestamp'].max()}\n"
                
            info += f"Colunas: {', '.join(self.data.columns[:10])}"
            if len(self.data.columns) > 10:
                info += f" e mais {len(self.data.columns) - 10} colunas"
        else:
            info = mensagem
            
        self.data_info_text.insert(tk.END, info)
        self.data_info_text.config(state=tk.DISABLED)
    
    def _add_technical_indicator(self):
        """Adiciona um indicador técnico aos dados"""
        if self.data is None:
            messagebox.showwarning("Aviso", "Carregue dados primeiro")
            return
            
        selected = self.indicator_type_var.get()
        
        # Extrair o código do indicador (parte antes do primeiro espaço)
        indicator_type = selected.split(' - ')[0]
        
        # Tratar indicadores avançados especiais
        advanced_indicators = ["ATR", "Parkinson", "Adaptive_RSI", "Force_Index"]
        
        if indicator_type in advanced_indicators:
            # Chamar método para indicadores avançados
            self._add_advanced_indicator(indicator_type)
            return
        
        # Processar indicadores técnicos padrão
        try:
            period = int(self.indicator_period_var.get())
            if period <= 0:
                messagebox.showwarning("Aviso", "O período deve ser maior que zero")
                return
        except ValueError:
            messagebox.showwarning("Aviso", "O período deve ser um número inteiro")
            return
            
        # Encontrar coluna de preço para aplicar o indicador
        price_column = self._get_price_column()
        if not price_column:
            messagebox.showwarning("Aviso", "Não foi possível identificar uma coluna de preço")
            return
        
        try:
            # Adicionar o indicador correspondente
            if indicator_type == "SMA":
                self.data[f'SMA_{period}'] = self.data[price_column].rolling(window=period).mean()
                indicator_name = f"SMA_{period}"
                
            elif indicator_type == "EMA":
                self.data[f'EMA_{period}'] = self.data[price_column].ewm(span=period, adjust=False).mean()
                indicator_name = f"EMA_{period}"
                
            elif indicator_type == "RSI":
                delta = self.data[price_column].diff()
                gain = delta.where(delta > 0, 0).rolling(window=period).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
                rs = gain / loss
                self.data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
                indicator_name = f"RSI_{period}"
                
            elif indicator_type == "MACD":
                ema12 = self.data[price_column].ewm(span=12, adjust=False).mean()
                ema26 = self.data[price_column].ewm(span=26, adjust=False).mean()
                self.data['MACD'] = ema12 - ema26
                self.data['MACD_Signal'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
                indicator_name = "MACD"
                
            elif indicator_type == "Bollinger Bands":
                sma = self.data[price_column].rolling(window=period).mean()
                std = self.data[price_column].rolling(window=period).std()
                self.data['BB_Upper'] = sma + (std * 2)
                self.data['BB_Middle'] = sma
                self.data['BB_Lower'] = sma - (std * 2)
                indicator_name = "Bollinger Bands"
                
            else:
                messagebox.showwarning("Aviso", "Indicador não implementado")
                return
                
            # Adicionar à lista de indicadores
            self.indicators_list.insert(tk.END, f"{indicator_name} ({price_column})")
            
            # Remover linhas com NaN criadas pelos indicadores
            self.data.dropna(inplace=True)
            
            # Atualizar informações e listas
            self._atualizar_info_dados("Dados atualizados com o novo indicador")
            self._atualizar_listas_features()
            
            self.status_var.set(f"Indicador {indicator_name} adicionado com sucesso")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao adicionar indicador: {str(e)}")
    
    def _get_price_column(self):
        """Identifica a coluna de preço para usar nos indicadores"""
        if self.data is None:
            return None
            
        # Lista de possíveis nomes de colunas de preço de fechamento
        candidates = ['close', 'Close', 'fechamento', 'Fechamento', 'price', 'Price']
        
        for col in candidates:
            if col in self.data.columns:
                return col
                
        # Se não encontrar, usar a primeira coluna numérica
        for col in self.data.columns:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                return col
                
        return None
    
    def _remove_technical_indicator(self):
        """Remove o indicador técnico selecionado"""
        if self.data is None or not self.indicators_list.curselection():
            return
            
        selected_index = self.indicators_list.curselection()[0]
        indicator_text = self.indicators_list.get(selected_index)
        
        # Extrair o nome do indicador
        indicator_name = indicator_text.split('(')[0].strip()
        
        # Encontrar colunas relacionadas a este indicador
        columns_to_drop = []
        for col in self.data.columns:
            if indicator_name in col:
                columns_to_drop.append(col)
                
        # Caso especial para Bollinger Bands
        if "Bollinger Bands" in indicator_name:
            for prefix in ['BB_Upper', 'BB_Middle', 'BB_Lower']:
                if prefix in self.data.columns:
                    columns_to_drop.append(prefix)
                    
        # Caso especial para MACD
        if "MACD" in indicator_name:
            if 'MACD_Signal' in self.data.columns:
                columns_to_drop.append('MACD_Signal')
        
        # Remover as colunas
        if columns_to_drop:
            self.data.drop(columns=columns_to_drop, inplace=True)
            
            # Atualizar informações e listas
            self._atualizar_info_dados("Dados atualizados após remover indicador")
            self._atualizar_listas_features()
            
            # Remover da lista
            self.indicators_list.delete(selected_index)
            
            self.status_var.set(f"Indicador {indicator_name} removido com sucesso")
        else:
            messagebox.showwarning("Aviso", "Não foi possível encontrar as colunas do indicador")
    
    def _atualizar_listas_features(self):
        """Atualiza as listas de colunas disponíveis e selecionadas"""
        if self.data is None:
            return
            
        # Limpar listas atuais
        self.available_columns.delete(0, tk.END)
        self.selected_columns.delete(0, tk.END)
        
        # Obter colunas dos dados
        columns = list(self.data.columns)
        
        # Filtrar colunas não numéricas
        numeric_columns = []
        for col in columns:
            if pd.api.types.is_numeric_dtype(self.data[col]) and col != 'timestamp':
                numeric_columns.append(col)
                
        # Popular lista de colunas disponíveis
        for col in sorted(numeric_columns):
            self.available_columns.insert(tk.END, col)
    
    def _add_all_columns(self):
        """Adiciona todas as colunas disponíveis à lista de selecionadas"""
        for i in range(self.available_columns.size()):
            col = self.available_columns.get(i)
            if col not in self.selected_columns.get(0, tk.END):
                self.selected_columns.insert(tk.END, col)
    
    def _add_selected_columns(self):
        """Adiciona as colunas selecionadas à lista de selecionadas"""
        for i in self.available_columns.curselection():
            col = self.available_columns.get(i)
            if col not in self.selected_columns.get(0, tk.END):
                self.selected_columns.insert(tk.END, col)
    
    def _remove_selected_columns(self):
        """Remove as colunas selecionadas da lista de selecionadas"""
        items = self.selected_columns.curselection()
        for i in reversed(items):  # Reverse para não afetar os índices durante a remoção
            self.selected_columns.delete(i)
    
    def _remove_all_columns(self):
        """Remove todas as colunas da lista de selecionadas"""
        self.selected_columns.delete(0, tk.END)
    
    def _treinar_modelo(self):
        """Treina o modelo com os parâmetros configurados"""
        if self.data is None:
            messagebox.showwarning("Aviso", "Carregue dados antes de treinar o modelo")
            return
            
        # Verificar se há colunas selecionadas
        if self.selected_columns.size() == 0:
            messagebox.showwarning("Aviso", "Selecione pelo menos uma coluna para treinamento")
            return
            
        # Iniciar treinamento em thread separada
        self.progress.start()
        self.train_button.config(state=tk.DISABLED)
        self.status_var.set("Treinando modelo...")
        
        threading.Thread(target=self._thread_treinamento).start()
    
    def _thread_treinamento(self):
        """Thread para treinamento do modelo em background"""
        try:
            # Preparar os dados
            df = self.data.copy()
            
            # Obter features selecionadas
            feature_cols = list(self.selected_columns.get(0, tk.END))
            self.selected_features = feature_cols
            
            # Preparar alvo (target) - o que queremos prever
            target_type = self.target_type_var.get()
            
            if target_type == "next_candle":
                price_col = self._get_price_column()
                if price_col:
                    df['target'] = (df[price_col].shift(-1) > df[price_col]).astype(int)
                else:
                    raise ValueError("Não foi possível identificar a coluna de preço")
            else:  # tendência
                periods = int(self.trend_periods_var.get())
                price_col = self._get_price_column()
                if price_col:
                    df['future_price'] = df[price_col].shift(-periods)
                    df['target'] = (df['future_price'] > df[price_col]).astype(int)
                    df.drop('future_price', axis=1, inplace=True)
                else:
                    raise ValueError("Não foi possível identificar a coluna de preço")
            
            # Remover linhas com valores NaN
            df.dropna(inplace=True)
            
            X = df[feature_cols]
            y = df['target']
            
            # Dividir dados em treino e teste
            train_size = float(self.train_size_var.get()) / 100
            split_method = self.split_method_var.get()
            
            if split_method == "time":
                # Divisão temporal
                split_idx = int(len(X) * train_size)
                X_train = X.iloc[:split_idx]
                X_test = X.iloc[split_idx:]
                y_train = y.iloc[:split_idx]
                y_test = y.iloc[split_idx:]
            elif split_method == "time_series_cv":
                # Validação cruzada temporal
                n_splits = int(self.n_splits_var.get())
                tscv = TimeSeriesSplit(n_splits=n_splits)
                splits = list(tscv.split(X))
                train_idx, test_idx = splits[-1]  # Usar o último split para treino e teste
                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
            else:
                # Divisão aleatória
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, train_size=train_size, random_state=42
                )

            # Aplicar balanceamento se configurado
            if hasattr(self, 'balance_method_var') and self.balance_method_var.get() != "none":
                balance_method = self.balance_method_var.get()
                
                try:
                    if balance_method == "undersample":
                        from imblearn.under_sampling import RandomUnderSampler
                        sampler = RandomUnderSampler(random_state=42)
                        if isinstance(X_train, np.ndarray):
                            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
                            X_train = X_train_resampled
                            y_train = y_train_resampled
                        else:
                            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train.values, y_train.values)
                            X_train = pd.DataFrame(X_train_resampled, columns=X_train.columns)
                            y_train = pd.Series(y_train_resampled)
                            
                    elif balance_method == "oversample":
                        from imblearn.over_sampling import RandomOverSampler
                        sampler = RandomOverSampler(random_state=42)
                        if isinstance(X_train, np.ndarray):
                            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
                            X_train = X_train_resampled
                            y_train = y_train_resampled
                        else:
                            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train.values, y_train.values)
                            X_train = pd.DataFrame(X_train_resampled, columns=X_train.columns)
                            y_train = pd.Series(y_train_resampled)
                            
                    elif balance_method == "SMOTE":
                        from imblearn.over_sampling import SMOTE
                        sampler = SMOTE(random_state=42)
                        if isinstance(X_train, np.ndarray):
                            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
                            X_train = X_train_resampled
                            y_train = y_train_resampled
                        else:
                            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train.values, y_train.values)
                            X_train = pd.DataFrame(X_train_resampled, columns=X_train.columns)
                            y_train = pd.Series(y_train_resampled)
                    
                    # Calcular proporção após balanceamento
                    if isinstance(y_train, np.ndarray):
                        class_counts = np.bincount(y_train)
                    else:
                        class_counts = y_train.value_counts().values
                        
                    class_ratio = class_counts[1] / class_counts[0] if len(class_counts) > 1 else 1.0
                    
                    # Registrar informação sobre balanceamento
                    self.balancing_info = {
                        'method': balance_method,
                        'before': None,  # precisaria calcular antes
                        'after': class_ratio,
                        'class_counts': class_counts.tolist() if isinstance(class_counts, np.ndarray) else class_counts
                    }
                    
                except ImportError:
                    self.balancing_info = {'error': 'Biblioteca imblearn não encontrada. Instale com: pip install imbalanced-learn'}
            
            
            # Normalizar dados se solicitado
            if self.normalize_var.get():
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                self.scaler = scaler
            else:
                self.scaler = None
                
            # Manter os dados para uso posterior
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            
            # Expandir opções de algoritmos com parâmetros otimizados para opções binárias
            algorithm = self.algorithm_var.get().split(" - ")[0]
            
            if algorithm == "LogisticRegression":
                model = LogisticRegression(
                    C=0.1,                  # Mais regularização para evitar overfitting
                    class_weight='balanced', # Corrigir desbalanceamento de classes
                    max_iter=1000,
                    penalty='l2',
                    solver='liblinear',     # Melhor para datasets pequenos/médios
                    random_state=42
                )
                param_grid = {
                    'C': [0.01, 0.05, 0.1, 0.5, 1.0], 
                    'solver': ['liblinear', 'saga'],
                    'class_weight': ['balanced', None]
                }
            elif algorithm == "RandomForest":
                model = RandomForestClassifier(
                    n_estimators=200,        # Mais árvores para maior estabilidade
                    max_depth=15,            # Limitar profundidade para evitar overfitting
                    min_samples_split=10,    # Exigir mais amostras para divisões
                    class_weight='balanced_subsample', # Balanceamento em cada subsample
                    random_state=42
                )
                
                # MODIFICADO: Espaço de parâmetros reduzido para RandomForest
                if self.tuning_var.get() and self.tuning_method_var.get() == "grid":
                    # Para Grid Search, usar um espaço de parâmetros muito reduzido
                    param_grid = {
                        'n_estimators': [100, 200],
                        'max_depth': [10, 15],
                        'min_samples_split': [10]
                    }
                elif self.tuning_var.get():
                    # Para Random Search, podemos manter mais opções
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [10, 15, 20],  # Remover None que é muito custoso
                        'min_samples_split': [5, 10, 15],
                        'class_weight': ['balanced', 'balanced_subsample']  # Remover None
                    }
                else:
                    param_grid = {}
                    
            elif algorithm == "GradientBoosting":
                model = GradientBoostingClassifier(
                    n_estimators=300,        # Mais estimadores
                    learning_rate=0.05,      # Taxa de aprendizado menor para melhor generalização
                    max_depth=5,             # Menor profundidade para evitar overfitting
                    subsample=0.8,           # Subamostragem para reduzir overfitting
                    random_state=42
                )
                param_grid = {
                    'n_estimators': [200, 300, 500],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.7, 0.8, 0.9]
                }
            elif algorithm == "XGBoost":
                try:
                    import xgboost as xgb
                    model = xgb.XGBClassifier(
                        objective='binary:logistic',
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=5,
                        subsample=0.8,
                        colsample_bytree=0.8,  # Usar apenas 80% das features em cada árvore
                        reg_alpha=0.1,         # Regularização L1
                        reg_lambda=1.0,        # Regularização L2
                        scale_pos_weight=1.0,  # Balanceamento de classes
                        random_state=42,
                        eval_metric='logloss'
                    )
                    param_grid = {
                        'n_estimators': [200, 300, 500],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.7, 0.8, 0.9],
                        'colsample_bytree': [0.7, 0.8, 0.9],
                        'reg_alpha': [0.0, 0.1, 0.5],
                        'scale_pos_weight': [1.0, 2.0, 3.0]  # Para dados desbalanceados
                    }
                except ImportError:
                    messagebox.showwarning("Aviso", "XGBoost não está instalado. Usando RandomForest.")
                    model = RandomForestClassifier(n_estimators=200, random_state=42)
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [10, 15, 20, None]
                    }
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                param_grid = {}

            # Após definir o modelo base e antes da otimização de hiperparâmetros

            # MODIFICADO: Otimização de hiperparâmetros com timeout e progresso
            if self.tuning_var.get():
                try:
                    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
                    
                    cv = int(self.tuning_cv_var.get())
                    
                    # Atualizar mensagem de status
                    self.frame.after(0, lambda: self.status_var.set("Iniciando otimização de hiperparâmetros..."))
                    
                    # Define timeout para cada ajuste (em segundos)
                    timeout = 60  # 1 minuto por combinação
                    
                    # Mostrar mensagem na área de resultados
                    self.frame.after(0, lambda: self._update_results_during_tuning("Otimização de hiperparâmetros iniciada.\n\nIsso pode levar vários minutos dependendo da complexidade do modelo e do espaço de busca.\n\nPor favor, aguarde..."))
                    
                    # Verificar se o algoritmo é muito pesado para Grid Search
                    if self.tuning_method_var.get() == "grid" and algorithm in ["RandomForest", "GradientBoosting", "XGBoost"]:
                        # NOVO: Aviso para algoritmos pesados
                        self.frame.after(0, lambda: messagebox.showinfo(
                            "Otimização pode ser lenta", 
                            "Grid Search com modelos complexos pode demorar muito tempo. "
                            "Considere usar Random Search para um processo mais rápido."
                        ))
                    
                    if self.tuning_method_var.get() == "random":
                        n_iter = int(self.tuning_iter_var.get())
                        search = RandomizedSearchCV(
                            model, param_grid, n_iter=n_iter, cv=cv,
                            scoring='accuracy', random_state=42, 
                            n_jobs=4,  # MODIFICADO: limitar o número de jobs
                            error_score='raise'
                        )
                    else:
                        search = GridSearchCV(
                            model, param_grid, cv=cv,
                            scoring='accuracy', 
                            n_jobs=4,  # MODIFICADO: limitar o número de jobs
                            error_score='raise'
                        )
                    
                    # Treinar com busca de hiperparâmetros
                    search.fit(X_train, y_train)
                    model = search.best_estimator_
                    
                    # Guardar os melhores parâmetros
                    self.best_params = search.best_params_
                    
                except Exception as e:
                    self.frame.after(0, lambda: messagebox.showwarning(
                        "Aviso na Otimização", 
                        f"Erro durante a otimização: {str(e)}\n"
                        "Continuando com os parâmetros padrão."
                    ))
                    model.fit(X_train, y_train)
                    self.best_params = None
            else:
                # Treinar normalmente
                model.fit(X_train, y_train)
                self.best_params = None
            
            self.model = model
            
            # Avaliar o modelo
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
            
            train_acc = accuracy_score(y_train, train_preds)
            test_acc = accuracy_score(y_test, test_preds)
            
            # Mais métricas para teste
            precision = precision_score(y_test, test_preds)
            recall = recall_score(y_test, test_preds)
            f1 = f1_score(y_test, test_preds)
            
            # Guardar previsões para visualização
            self.predictions = test_preds
            
            # Preparar mensagem de resultados
            results = (
                "=== Resultados do Treinamento ===\n\n"
                f"Modelo: {algorithm}\n"
                f"Features usadas: {len(feature_cols)}\n"
                f"Dados: {len(X)} registros (Treino: {len(X_train)}, Teste: {len(X_test)})\n\n"
                f"Acurácia no Treino: {train_acc:.4f} ({train_acc*100:.1f}%)\n"
                f"Acurácia no Teste: {test_acc:.4f} ({test_acc*100:.1f}%)\n\n"
                f"Precisão: {precision:.4f}\n"
                f"Recall: {recall:.4f}\n"
                f"F1-Score: {f1:.4f}\n\n"
                f"Melhores parâmetros: {self.best_params}\n\n"
                "O modelo está pronto para fazer previsões!"
            )
            
            # Atualizar interface no thread principal
            self.frame.after(0, lambda: self._atualizar_apos_treino(results, test_acc))
                
        except Exception as e:
            # Substitua self.master.after por self.frame.after
            self.frame.after(0, lambda: self._erro_treino(str(e)))
            import traceback
            traceback.print_exc()
    
    def _update_results_during_tuning(self, message):
        """Atualiza a área de resultados durante o tuning"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert(tk.END, message)
        self.results_text.config(state=tk.DISABLED)
    
    def _atualizar_apos_treino(self, results, accuracy):
        """Atualiza a interface após treinamento bem-sucedido"""
        # Parar indicador de progresso
        self.progress.stop()
        self.train_button.config(state=tk.NORMAL)
        self.status_var.set(f"Treinamento concluído com acurácia de {accuracy:.4f}")
        
        # Mostrar resultados
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert(tk.END, results)
        self.results_text.config(state=tk.DISABLED)
        
        # Atualizar informações do modelo na aba de previsão
        timeframe = self.timeframe_var.get()
        self.model_status_var.set(f"Modelo: {self.algorithm_var.get().split(' - ')[0]} ({timeframe})")
        self.model_accuracy_var.set(f"Acurácia: {accuracy:.4f} ({accuracy*100:.1f}%)")
        
        # Trocar para a aba de resultados
        self.notebook.select(2)  # Índice 2 é a aba "Resultados"
    
    def _erro_treino(self, erro):
        """Trata erros durante o treinamento"""
        self.progress.stop()
        self.train_button.config(state=tk.NORMAL)
        self.status_var.set(f"Erro no treinamento: {erro}")
        messagebox.showerror("Erro no Treinamento", f"Ocorreu um erro: {erro}")
    
    def _plot_confusion_matrix(self):
        """Plota a matriz de confusão do modelo"""
        if not hasattr(self, 'model') or self.model is None:
            messagebox.showwarning("Aviso", "Treine um modelo primeiro")
            return
            
        if not hasattr(self, 'y_test') or not hasattr(self, 'predictions'):
            messagebox.showwarning("Aviso", "Dados de teste não disponíveis")
            return
        
        # Limpar frame do gráfico
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        # Calcular matriz de confusão
        cm = confusion_matrix(self.y_test, self.predictions)
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Plotar matriz de confusão
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Rótulos
        classes = ["Baixa (0)", "Alta (1)"]
        ax.set(xticks=[0, 1], yticks=[0, 1],
              xticklabels=classes, yticklabels=classes,
              ylabel='Valor Real',
              xlabel='Valor Previsto')
        
        # Adicionar valores na matriz
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", 
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14)
        
        ax.set_title("Matriz de Confusão", fontsize=14)
        plt.tight_layout()
        
        # Mostrar na interface
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)
        
        # Adicionar barra de navegação
        toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
        toolbar.update()
    
    def _plot_feature_importance(self):
        """Plota a importância das features para o modelo"""
        if not hasattr(self, 'model') or self.model is None:
            messagebox.showwarning("Aviso", "Treine um modelo primeiro")
            return
            
        # Verificar se o modelo tem atributo de importância de features
        if not hasattr(self.model, 'feature_importances_'):
            messagebox.showinfo("Informação", 
                              "Este modelo não fornece importância de features.\n"
                              "Use RandomForest ou GradientBoosting para ver importância de features.")
            return
            
        # Limpar frame do gráfico
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        # Preparar dados
        features = self.selected_features
        importances = self.model.feature_importances_
        
        # Ordenar por importância
        indices = np.argsort(importances)[::-1]
        sorted_features = [features[i] for i in indices]
        sorted_importances = [importances[i] for i in indices]
        
        # Limitar a 15 features para melhor visualização
        if len(sorted_features) > 15:
            sorted_features = sorted_features[:15]
            sorted_importances = sorted_importances[:15]
            
        # Criar figura
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plotar barras
        bars = ax.barh(range(len(sorted_features)), sorted_importances, align='center')
        
        # Colorir barras por gradiente
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(i/len(sorted_features)))
        
        # Configurar gráfico
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('Importância Relativa')
        ax.set_title('Importância das Features')
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
        # Mostrar na interface
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)
        
        # Adicionar barra de navegação
        toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
        toolbar.update()
    
    def _salvar_modelo(self):
        """Salva o modelo treinado"""
        if not hasattr(self, 'model') or self.model is None:
            messagebox.showwarning("Aviso", "Treine um modelo primeiro")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Salvar Modelo",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("Joblib files", "*.joblib"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # Criar um dicionário com todas as informações necessárias
            model_data = {
                'model': self.model,
                'features': self.selected_features,
                'scaler': self.scaler,
                'algorithm': self.algorithm_var.get().split(" - ")[0],
                'accuracy': accuracy_score(self.y_test, self.predictions),
                'date_trained': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'timeframe': self.timeframe_var.get(),  # NOVO: Salvar o timeframe
                'target_type': self.target_type_var.get(),  # NOVO: Salvar o tipo de alvo
                'trend_periods': self.trend_periods_var.get() if self.target_type_var.get() == "trend" else "1",
                'best_params': self.best_params  # NOVO: Salvar os melhores parâmetros
            }
            
            # Salvar dependendo da extensão
            if file_path.endswith('.joblib'):
                joblib.dump(model_data, file_path)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(model_data, f)
                    
            messagebox.showinfo("Sucesso", f"Modelo salvo em {file_path}")
            self.status_var.set(f"Modelo salvo em {file_path}")
            
        except Exception as e:
            messagebox.showerror("Erro ao salvar", f"Erro: {str(e)}")
            self.status_var.set(f"Erro ao salvar modelo: {str(e)}")
        
    def _carregar_modelo(self):
        """Carrega um modelo salvo anteriormente"""
        file_path = filedialog.askopenfilename(
            title="Carregar Modelo",
            filetypes=[("Pickle files", "*.pkl"), ("Joblib files", "*.joblib"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # Carregar dependendo da extensão
            if file_path.endswith('.joblib'):
                model_data = joblib.load(file_path)
            else:
                with open(file_path, 'rb') as f:
                    model_data = pickle.load(f)
                    
            # Extrair os dados
            self.model = model_data['model']
            self.selected_features = model_data.get('features', [])
            self.scaler = model_data.get('scaler', None)
            
            # Atualizar informações do modelo
            algorithm = model_data.get('algorithm', type(self.model).__name__)
            accuracy = model_data.get('accuracy', 0)
            date_trained = model_data.get('date_trained', 'desconhecida')
            timeframe = model_data.get('timeframe', '?')  # NOVO: Recuperar o timeframe
            target_type = model_data.get('target_type', 'next_candle')  # NOVO: Recuperar o tipo de alvo
            self.best_params = model_data.get('best_params', None)  # NOVO: Recuperar os melhores parâmetros
            
            # Atualizar variáveis de interface para refletir o modelo carregado
            self.timeframe_var.set(timeframe)
            self.target_type_var.set(target_type)
            if 'trend_periods' in model_data and target_type == "trend":
                self.trend_periods_var.set(model_data['trend_periods'])
            
            # Atualizar interface
            self.model_status_var.set(f"Modelo: {algorithm} ({timeframe}) (carregado de arquivo)")
            self.model_accuracy_var.set(f"Acurácia: {accuracy:.4f} ({accuracy*100:.1f}%)")
            
            # Atualizar a informação na aba de treinamento
            self.model_info_var.set(f"Modelo atual: {algorithm} ({timeframe}) (Acurácia: {accuracy:.4f})")
            
            # Mostrar mensagem nos resultados
            target_info = "Próxima vela" if target_type == "next_candle" else f"Tendência ({model_data.get('trend_periods', '?')} períodos)"
            
            results = (
                f"Modelo Carregado: {algorithm}\n"
                f"Timeframe: {timeframe}\n"
                f"Tipo de previsão: {target_info}\n"
                f"Data de treinamento: {date_trained}\n"
                f"Acurácia reportada: {accuracy:.4f} ({accuracy*100:.1f}%)\n\n"
                f"Melhores parâmetros: {self.best_params}\n\n"
                f"Features usadas ({len(self.selected_features)}):\n"
                f"{', '.join(self.selected_features[:10])}"
                f"{' e mais...' if len(self.selected_features) > 10 else ''}\n\n"
                "O modelo está pronto para fazer previsões!"
            )
            
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete('1.0', tk.END)
            self.results_text.insert(tk.END, results)
            self.results_text.config(state=tk.DISABLED)
            
            messagebox.showinfo("Sucesso", f"Modelo carregado de {file_path}")
            self.status_var.set(f"Modelo carregado com sucesso")
            
            # Trocar para a aba de resultados
            self.notebook.select(2)  # Índice 2 é a aba "Resultados"
            
        except Exception as e:
            messagebox.showerror("Erro ao carregar", f"Erro: {str(e)}")
            self.status_var.set(f"Erro ao carregar modelo: {str(e)}")
    
    def _fazer_previsao(self):
        """Faz uma previsão com o modelo treinado"""
        if not hasattr(self, 'model') or self.model is None:
            messagebox.showwarning("Aviso", "Treine ou carregue um modelo primeiro")
            return
            
        source = self.prediction_source_var.get()
        
        if source == "test" and hasattr(self, 'X_test'):
            # Usar dados de teste
            self._mostrar_previsao_teste()
        elif source == "recent":
            # Buscar dados recentes
            self._fazer_previsao_recente()
        else:
            messagebox.showwarning("Aviso", "Dados de teste não disponíveis. Treine um modelo primeiro.")
    
    def _mostrar_previsao_teste(self):
        """Mostra as previsões nos dados de teste"""
        # Verificar disponibilidade dos dados
        if not hasattr(self, 'X_test') or not hasattr(self, 'y_test'):
            messagebox.showwarning("Aviso", "Dados de teste não disponíveis")
            return
            
        try:
            # Obter número de candles
            try:
                num_candles = min(int(self.prediction_candles_var.get()), len(self.X_test))
            except ValueError:
                num_candles = min(10, len(self.X_test))
                
            # Fazer predição
            X_subset = self.X_test[:num_candles] if hasattr(self.X_test, '__getitem__') else self.X_test
            y_subset = self.y_test[:num_candles]
            
            # Fazer previsão
            predictions = self.model.predict(X_subset)
            probabilities = None
            
            # Tentar obter probabilidades (nem todos modelos suportam)
            if hasattr(self.model, 'predict_proba'):
                try:
                    probabilities = self.model.predict_proba(X_subset)
                except:
                    pass
                    
            # Montar resultados para exibição
            result_text = "=== Previsão nos Dados de Teste ===\n\n"
            result_text += f"Usando {num_candles} candles dos dados de teste\n\n"
            result_text += f"{'#':<3} {'Real':<8} {'Previsto':<8} {'Confiança':<10} {'Correto?':<8}\n"
            result_text += "-" * 40 + "\n"
            
            correct_count = 0
            
            for i in range(len(predictions)):
                real = y_subset.iloc[i] if hasattr(y_subset, 'iloc') else y_subset[i]
                pred = predictions[i]
                
                # Determinar confiança (probabilidade)
                confidence = "N/A"
                if probabilities is not None:
                    confidence = f"{probabilities[i][pred]:.2f}"
                    
                # Verificar se está correto
                is_correct = real == pred
                if is_correct:
                    correct_count += 1
                    
                result_text += f"{i+1:<3} {'Alta' if real == 1 else 'Baixa':<8} "
                result_text += f"{'Alta' if pred == 1 else 'Baixa':<8} "
                result_text += f"{confidence:<10} "
                result_text += f"{'✓' if is_correct else '✗'}\n"
                
            # Acurácia
            accuracy = correct_count / len(predictions) if len(predictions) > 0 else 0
            result_text += f"\nAcurácia: {accuracy:.2f} ({correct_count}/{len(predictions)})"
            
            # Mostrar resultados
            self.prediction_text.config(state=tk.NORMAL)
            self.prediction_text.delete('1.0', tk.END)
            self.prediction_text.insert(tk.END, result_text)
            self.prediction_text.config(state=tk.DISABLED)
            
            # Plotar gráfico comparativo
            self._plot_prediction_comparison(y_subset, predictions)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao fazer previsão: {str(e)}")
            import traceback
            traceback.print_exc()

    def _fazer_previsao_recente(self):
        """Faz previsão usando os dados mais recentes da coleta"""
        if not self.app or not hasattr(self.app, 'data_collection_tab'):
            messagebox.showwarning("Aviso", "Aba de coleta de dados não disponível")
            return
            
        if not hasattr(self.app.data_collection_tab, 'collected_data'):
            messagebox.showwarning("Aviso", "Nenhum dado disponível na aba de coleta")
            return
            
        try:
            # Obter dados recentes
            recent_data = self.app.data_collection_tab.collected_data.copy()
            
            # Verificar se temos o número suficiente de registros
            if recent_data.shape[0] == 0:
                messagebox.showwarning("Aviso", "Não há dados recentes disponíveis")
                return
                
            # Obter número de candles para previsão
            try:
                num_candles = min(int(self.prediction_candles_var.get()), recent_data.shape[0])
            except ValueError:
                num_candles = min(10, recent_data.shape[0])
                
            # Usar apenas as colunas que o modelo conhece
            features_df = recent_data.copy()
            
            # Verificar quais features do modelo estão disponíveis nos dados recentes
            missing_features = []
            for feature in self.selected_features:
                if feature not in features_df.columns:
                    missing_features.append(feature)
                    
            if missing_features:
                error_msg = f"Dados recentes não contêm todas as features necessárias. Faltando:\n{', '.join(missing_features)}"
                messagebox.showerror("Erro", error_msg)
                return
                
            # Selecionar apenas as features usadas no modelo
            X_recent = features_df[self.selected_features].tail(num_candles)
            
            # Aplicar normalização se necessário
            if self.scaler is not None:
                X_recent_scaled = self.scaler.transform(X_recent)
                predictions = self.model.predict(X_recent_scaled)
            else:
                predictions = self.model.predict(X_recent)
                
            # Tentar obter probabilidades
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                try:
                    if self.scaler is not None:
                        probabilities = self.model.predict_proba(X_recent_scaled)
                    else:
                        probabilities = self.model.predict_proba(X_recent)
                except:
                    pass
                    
            # Tentar determinar os valores reais para comparação (quando possível)
            has_real_values = False
            real_values = None
            accuracy_info = ""
            
            # Se temos dados de preço, podemos calcular os resultados reais para comparação
            price_col = self._get_price_column()
            if price_col and len(recent_data) > num_candles + 1:
                # Calcular se o preço subiu ou desceu na próxima vela
                future_prices = recent_data[price_col].shift(-1).tail(num_candles)
                current_prices = recent_data[price_col].tail(num_candles)
                
                # Remover NaN (a última vela não terá valor futuro)
                mask = ~future_prices.isna()
                real_values = (future_prices > current_prices).astype(int)
                has_real_values = True
                
                # Calcular acurácia para os valores disponíveis
                correct_count = 0
                total_valid = 0
                
                for i in range(len(predictions)):
                    if i < len(real_values) and not pd.isna(real_values.iloc[i]):
                        if predictions[i] == real_values.iloc[i]:
                            correct_count += 1
                        total_valid += 1
                
                if total_valid > 0:
                    accuracy = correct_count / total_valid
                    accuracy_info = f"\nAcurácia: {accuracy:.2f} ({correct_count}/{total_valid})"
            
            # Montar resultados para exibição
            result_text = "=== Previsão em Dados Recentes ===\n\n"
            result_text += f"Usando {num_candles} candles dos dados recentes\n\n"
            
            # Incluir timestamps se disponíveis
            has_timestamp = False
            timestamps = []
            
            if 'timestamp' in recent_data.columns:
                timestamps = recent_data['timestamp'].tail(num_candles).tolist()
                has_timestamp = True
                if has_real_values:
                    result_text += f"{'Data/Hora':<20} {'Previsão':<10} {'Confiança':<10} {'Real':<10} {'Correto?':<8}\n"
                else:
                    result_text += f"{'Data/Hora':<20} {'Previsão':<10} {'Confiança':<10}\n"
            else:
                if has_real_values:
                    result_text += f"{'#':<5} {'Previsão':<10} {'Confiança':<10} {'Real':<10} {'Correto?':<8}\n"
                else:
                    result_text += f"{'#':<5} {'Previsão':<10} {'Confiança':<10}\n"
                    
            result_text += "-" * (60 if has_real_values else 40) + "\n"
            
            for i in range(len(predictions)):
                pred = predictions[i]
                
                # Determinar confiança
                confidence = "N/A"
                if probabilities is not None:
                    confidence = f"{probabilities[i][pred]:.2f}"
                    
                if has_timestamp:
                    ts = timestamps[i]
                    if isinstance(ts, str):
                        time_str = ts[:19]  # Truncar se for string longa
                    else:
                        time_str = str(ts)[:19]
                    result_text += f"{time_str:<20} "
                else:
                    result_text += f"{i+1:<5} "
                    
                result_text += f"{'Alta' if pred == 1 else 'Baixa':<10} "
                result_text += f"{confidence:<10} "
                
                # Adicionar informação de acerto/erro quando disponível
                if has_real_values and i < len(real_values) and not pd.isna(real_values.iloc[i]):
                    real_val = real_values.iloc[i]
                    is_correct = pred == real_val
                    result_text += f"{'Alta' if real_val == 1 else 'Baixa':<10} "
                    result_text += f"{'✓' if is_correct else '✗'}\n"
                else:
                    result_text += "\n" if not has_real_values else "N/A        N/A\n"
                    
            # Adicionar informação de acurácia se disponível
            result_text += accuracy_info

            # REMOVIDO: Seção de threshold e decisões de trading
                
            # Mostrar resultados
            self.prediction_text.config(state=tk.NORMAL)
            self.prediction_text.delete('1.0', tk.END)
            self.prediction_text.insert(tk.END, result_text)
            self.prediction_text.config(state=tk.DISABLED)
            
            # Plotar gráfico dos dados recentes com previsão
            self._plot_recent_prediction(recent_data.tail(num_candles), predictions, real_values if has_real_values else None)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao fazer previsão: {str(e)}")
            import traceback
            traceback.print_exc()

    def _plot_prediction_comparison(self, y_true, y_pred):
        """Plota um gráfico comparando valores reais e previstos"""
        # Limpar frame do gráfico
        for widget in self.prediction_plot_frame.winfo_children():
            widget.destroy()
            
        # Converter para arrays
        if hasattr(y_true, 'values'):
            y_true = y_true.values
        if hasattr(y_pred, 'values'):
            y_pred = y_pred.values
            
        # Criar figura
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Criar dados para o eixo X
        x = np.arange(len(y_true))
        
        # Plotar valores reais e previstos
        ax.plot(x, y_true, 'o-', label='Valor Real', color='blue', markersize=8)
        ax.plot(x, y_pred, 'o--', label='Previsão', color='red', markersize=6)
        
        # Configurar gráfico
        ax.set_title('Comparação: Valores Reais vs. Previstos')
        ax.set_ylabel('Classe (0=Baixa, 1=Alta)')
        ax.set_xlabel('Índice da Amostra')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Baixa', 'Alta'])
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Mostrar na interface
        canvas = FigureCanvasTkAgg(fig, master=self.prediction_plot_frame)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)
        
        # Adicionar barra de navegação
        toolbar = NavigationToolbar2Tk(canvas, self.prediction_plot_frame)
        toolbar.update()
        
    def _plot_recent_prediction(self, recent_data, predictions, real_values=None):
        """Plota um gráfico com os dados recentes e as previsões"""
        # Limpar frame do gráfico
        for widget in self.prediction_plot_frame.winfo_children():
            widget.destroy()
                
        try:
            # Identificar coluna de preço
            price_col = self._get_price_column()
                
            if price_col is None:
                raise ValueError("Não foi possível identificar a coluna de preço")
                    
            # Criar figura
            fig, ax = plt.subplots(figsize=(8, 4))
                
            # Dados para plotagem
            prices = recent_data[price_col].values
            x = np.arange(len(prices))
                
            # Plotar preço
            ax.plot(x, prices, 'o-', label='Preço', color='blue')
                
            # Adicionar marcadores de previsão
            for i, pred in enumerate(predictions):
                color = 'green' if pred == 1 else 'red'
                marker = '^' if pred == 1 else 'v'
                ax.scatter(i, prices[i], color=color, marker=marker, s=100,
                        label='Alta prevista' if pred == 1 else 'Baixa prevista')
            
            # Adicionar marcadores para valores reais, se disponíveis
            if real_values is not None:
                for i, real in enumerate(real_values):
                    if pd.isna(real):
                        continue
                        
                    edge_color = 'lime' if real == 1 else 'darkred'
                    ax.scatter(i, prices[i], facecolors='none', edgecolors=edge_color, 
                            s=150, linewidth=2, marker='o',
                            label='Real: Alta' if real == 1 else 'Real: Baixa')
                    
            # Remover duplicatas da legenda
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
                
            # Configurar gráfico
            ax.set_title('Previsão nos Dados Recentes')
            ax.set_ylabel('Preço')
            ax.set_xlabel('Índice da Amostra')
            ax.grid(True, linestyle='--', alpha=0.7)
                
            # Mostrar na interface
            canvas = FigureCanvasTkAgg(fig, master=self.prediction_plot_frame)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.pack(fill=tk.BOTH, expand=True)
                
            # Adicionar barra de navegação
            toolbar = NavigationToolbar2Tk(canvas, self.prediction_plot_frame)
            toolbar.update()
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao plotar dados recentes: {str(e)}")
            import traceback
            traceback.print_exc()

    def _continuar_treinamento(self):
        """Continua o treinamento de um modelo existente"""
        if not hasattr(self, 'model') or self.model is None:
            messagebox.showwarning("Aviso", "Não há modelo para continuar treinamento.\nCarregue um modelo primeiro.")
            return
        
        if self.data is None:
            messagebox.showwarning("Aviso", "Carregue dados para continuar o treinamento.")
            return
            
        # Verificar se há colunas selecionadas
        if self.selected_columns.size() == 0:
            messagebox.showwarning("Aviso", "Selecione pelo menos uma coluna para treinamento")
            return
            
        # Verificar compatibilidade de features
        selected_features = list(self.selected_columns.get(0, tk.END))
        
        missing_features = []
        for feature in self.selected_features:
            if feature not in selected_features:
                missing_features.append(feature)
        
        if missing_features:
            msg = f"Os dados atuais não contêm todas as features usadas pelo modelo.\nFaltando: {', '.join(missing_features)}"
            messagebox.showerror("Erro de Compatibilidade", msg)
            return
        
        # Confirmar com o usuário
        confirm = messagebox.askyesno(
            "Confirmar Treinamento Continuado", 
            "Ao continuar o treinamento, o modelo existente será atualizado com os novos dados. Continuar?"
        )
        
        if not confirm:
            return
        
        # Iniciar treinamento em thread separada
        self.progress.start()
        self.train_button.config(state=tk.DISABLED)
        self.status_var.set("Continuando treinamento do modelo...")
        
        # Usar uma thread para não congelar a interface
        threading.Thread(target=self._thread_continuar_treinamento).start()

    def _thread_continuar_treinamento(self):
        """Thread para continuação de treinamento em background com aprendizado incremental"""
        try:
            # Preparar os dados novos
            df = self.data.copy()
            
            # Usar apenas as features que o modelo já conhece
            df = df[self.selected_features + ['target'] if 'target' in df.columns else self.selected_features]
            
            # Preparar alvo (target) se não existir
            if 'target' not in df.columns:
                target_type = self.target_type_var.get()
                
                if target_type == "next_candle":
                    price_col = self._get_price_column()
                    if price_col:
                        df['target'] = (df[price_col].shift(-1) > df[price_col]).astype(int)
                    else:
                        raise ValueError("Não foi possível identificar a coluna de preço")
                else:  # tendência
                    periods = int(self.trend_periods_var.get())
                    price_col = self._get_price_column()
                    if price_col:
                        df['future_price'] = df[price_col].shift(-periods)
                        df['target'] = (df['future_price'] > df[price_col]).astype(int)
                        df.drop('future_price', axis=1, inplace=True)
                    else:
                        raise ValueError("Não foi possível identificar a coluna de preço")
            
            # Remover linhas com valores NaN
            df.dropna(inplace=True)
            
            # Preparar dados de treinamento
            X_new = df[self.selected_features]
            y_new = df['target']
            
            # Normalizar se necessário
            if self.normalize_var.get() and self.scaler is not None:
                X_new_scaled = self.scaler.transform(X_new)
            else:
                X_new_scaled = X_new
            
            # Dividir os dados para treino/teste
            train_size = float(self.train_size_var.get()) / 100
            split_method = self.split_method_var.get()
            
            if split_method == "time":
                # Divisão temporal
                split_idx = int(len(X_new) * train_size)
                X_train = X_new_scaled[:split_idx] if isinstance(X_new_scaled, np.ndarray) else X_new_scaled.iloc[:split_idx]
                X_test = X_new_scaled[split_idx:] if isinstance(X_new_scaled, np.ndarray) else X_new_scaled.iloc[split_idx:]
                y_train = y_new[:split_idx] if isinstance(y_new, np.ndarray) else y_new.iloc[:split_idx]
                y_test = y_new[split_idx:] if isinstance(y_new, np.ndarray) else y_new.iloc[split_idx:]
            else:
                # Divisão aleatória
                X_train, X_test, y_train, y_test = train_test_split(
                    X_new_scaled, y_new, train_size=train_size, random_state=42
                )
            
            # ===== IMPLEMENTAÇÃO DO APRENDIZADO INCREMENTAL =====
            self.frame.after(0, lambda: self.status_var.set("Continuando treinamento do modelo..."))
            
            # Determinar a estratégia de treinamento com base no tipo de modelo
            model_type = type(self.model).__name__
            
            # Aplicar otimização de hiperparâmetros se ativada
            if self.tuning_var.get():
                try:
                    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
                    
                    cv = int(self.tuning_cv_var.get())
                    
                    # Descobrir parâmetros atuais do modelo para definir espaço de busca
                    current_params = self.model.get_params()
                    
                    # Definir espaço de busca em torno dos parâmetros atuais
                    param_grid = self._get_param_grid_for_model(model_type, current_params)
                    
                    if param_grid:  # Se temos parâmetros para otimizar
                        self.frame.after(0, lambda: self.status_var.set(
                            "Otimizando hiperparâmetros para o treinamento continuado..."))
                        
                        if self.tuning_method_var.get() == "random":
                            n_iter = int(self.tuning_iter_var.get())
                            search = RandomizedSearchCV(
                                self.model, param_grid, n_iter=n_iter, cv=cv,
                                scoring='accuracy', random_state=42, 
                                n_jobs=4, error_score='raise'
                            )
                        else:
                            search = GridSearchCV(
                                self.model, param_grid, cv=cv,
                                scoring='accuracy', 
                                n_jobs=4, error_score='raise'
                            )
                        
                        # Treinar com busca de hiperparâmetros
                        search.fit(X_train, y_train)
                        self.model = search.best_estimator_
                        self.best_params = search.best_params_
                        
                        self.frame.after(0, lambda: self.status_var.set(
                            f"Hiperparâmetros otimizados: {self.best_params}"))
                    
                except Exception as e:
                    self.frame.after(0, lambda: messagebox.showwarning(
                        "Aviso na Otimização", 
                        f"Erro durante a otimização: {str(e)}\n"
                        "Continuando com os parâmetros atuais."
                    ))
            
            # Estratégias específicas por tipo de modelo
            if hasattr(self.model, "partial_fit"):
                # Modelos que suportam aprendizado incremental nativo
                self.frame.after(0, lambda: self.status_var.set("Usando aprendizado incremental nativo..."))
                unique_classes = np.unique(y_train)
                
                # Aplicar partial_fit em mini-batches para melhor aprendizado
                batch_size = 50
                for i in range(0, len(X_train), batch_size):
                    end = min(i + batch_size, len(X_train))
                    X_batch = X_train[i:end]
                    y_batch = y_train[i:end]
                    self.model.partial_fit(X_batch, y_batch, classes=unique_classes)
                    
                training_method = "aprendizado incremental (partial_fit)"
                
            elif model_type in ["RandomForestClassifier"]:
                # Aprendizado incremental para RandomForest
                self.frame.after(0, lambda: self.status_var.set("Adaptando Random Forest para aprendizado incremental..."))
                
                # Manter alguns estimadores anteriores e adicionar novos
                old_estimators = self.model.estimators_[:]
                n_old = len(old_estimators)
                
                # Criar novos estimadores com os novos dados
                new_model = type(self.model)(**self.model.get_params())
                new_model.fit(X_train, y_train)
                new_estimators = new_model.estimators_
                
                # Mesclar estimadores (substituir alguns antigos por novos)
                replace_ratio = 0.3  # Substituir 30% dos estimadores antigos
                n_replace = int(n_old * replace_ratio)
                
                keep_indices = np.random.choice(n_old, n_old - n_replace, replace=False)
                merged_estimators = [old_estimators[i] for i in keep_indices] + new_estimators[:n_replace]
                
                # Atualizar estimadores do modelo
                self.model.estimators_ = merged_estimators
                
                training_method = "substituição parcial de estimadores (Random Forest)"
                
            elif model_type in ["GradientBoostingClassifier", "XGBClassifier"]:
                # Para modelos de boosting, podemos continuar o boosting com novos dados
                self.frame.after(0, lambda: self.status_var.set("Continuando boosting com novos dados..."))
                
                # Salvar parâmetros atuais
                params = self.model.get_params()
                
                # Criar novo modelo com parâmetros similares e menor taxa de aprendizado
                decay_factor = 0.5  # Reduzir a taxa de aprendizado
                if 'learning_rate' in params:
                    params['learning_rate'] = params['learning_rate'] * decay_factor
                
                # Criar modelo para os novos dados
                if model_type == "XGBClassifier":
                    # Técnica especial para XGBoost
                    import xgboost as xgb
                    # Continuar treinamento com novos dados
                    new_rounds = 100  # Adicionar X novas rodadas de boosting
                    
                    if hasattr(self.model, 'get_booster'):
                        try:
                            # Converter dados para formato DMatrix
                            dtrain = xgb.DMatrix(X_train, label=y_train)
                            # Continuar boosting a partir do modelo existente
                            self.model = xgb.train(params, dtrain, new_rounds, xgb_model=self.model.get_booster())
                            training_method = "boosting continuado (XGBoost)"
                        except Exception as e:
                            # Fallback - treinar normalmente
                            self.model.fit(X_train, y_train)
                            training_method = "retreino completo (XGBoost)"
                    else:
                        # Retreinar normalmente se não conseguir fazer o boosting incremental
                        self.model.fit(X_train, y_train)
                        training_method = "retreino completo (XGBoost)"
                else:
                    # Para GradientBoosting padrão
                    # Criar uma nova instância com menos estimadores
                    new_model = type(self.model)(**params)
                    new_model.fit(X_train, y_train)
                    
                    # Adicionar os novos estimadores ao modelo original
                    self.model.estimators_ = np.append(self.model.estimators_, new_model.estimators_, axis=0)
                    
                    training_method = "boosting estendido"
                    
            else:
                # Para outros modelos, usar abordagem de retreinamento ponderado
                self.frame.after(0, lambda: self.status_var.set("Usando retreinamento adaptativo..."))
                
                # Aumentar a importância dos dados novos no treinamento
                sample_weights = None
                if hasattr(self.model, 'fit') and 'sample_weight' in self.model.fit.__code__.co_varnames:
                    # Usar pesos de amostra para dar mais importância aos dados novos
                    sample_weights = np.ones(len(X_train))
                    # Dar 50% mais importância para os dados mais recentes
                    recent_portion = 0.3  # últimos 30% dos dados
                    recent_start = int(len(X_train) * (1 - recent_portion))
                    sample_weights[recent_start:] = 1.5
                
                # Treinar com pesos se disponível
                if sample_weights is not None:
                    self.model.fit(X_train, y_train, sample_weight=sample_weights)
                    training_method = "retreino com pesos adaptativos"
                else:
                    self.model.fit(X_train, y_train)
                    training_method = "retreino completo"
            
            # Manter os dados de teste para avaliação
            self.X_test = X_test
            self.y_test = y_test
            
            # Avaliar o modelo atualizado
            train_preds = self.model.predict(X_train)
            test_preds = self.model.predict(X_test)
            
            train_acc = accuracy_score(y_train, train_preds)
            test_acc = accuracy_score(y_test, test_preds)
            
            # Mais métricas para teste
            precision = precision_score(y_test, test_preds)
            recall = recall_score(y_test, test_preds)
            f1 = f1_score(y_test, test_preds)
            
            # Guardar previsões para visualização
            self.predictions = test_preds
            
            # Preparar mensagem de resultados
            results = (
                "=== Resultados do Treinamento Continuado ===\n\n"
                f"Modelo: {type(self.model).__name__}\n"
                f"Features usadas: {len(self.selected_features)}\n"
                f"Método: {training_method}\n"
                f"Dados novos: {len(X_train) + len(X_test)} registros\n"
                f"- Treino: {len(X_train)} exemplos\n"
                f"- Teste: {len(X_test)} exemplos\n\n"
                f"Acurácia no Treino: {train_acc:.4f} ({train_acc*100:.1f}%)\n"
                f"Acurácia no Teste: {test_acc:.4f} ({test_acc*100:.1f}%)\n\n"
                f"Precisão: {precision:.4f}\n"
                f"Recall: {recall:.4f}\n"
                f"F1-Score: {f1:.4f}\n\n"
                "O modelo evoluiu com novos dados, preservando o conhecimento anterior!"
            )
            
            # Atualizar interface no thread principal
            self.frame.after(0, lambda: self._atualizar_apos_treino(results, test_acc))
            
        except Exception as e:
            self.frame.after(0, lambda: self._erro_treino(str(e)))
            import traceback
            traceback.print_exc()
    
    def _get_param_grid_for_model(self, model_type, current_params):
        """Define espaços de parâmetros para otimização com base nos parâmetros atuais"""
        param_grid = {}
        
        if model_type == "LogisticRegression":
            # Explorar em torno do valor atual de C
            current_c = current_params.get('C', 1.0)
            param_grid = {
                'C': [current_c * 0.5, current_c, current_c * 1.5],
                'class_weight': ['balanced', None]
            }
            
        elif model_type == "RandomForestClassifier":
            # Parâmetros para RandomForest
            current_max_depth = current_params.get('max_depth', None)
            if current_max_depth is None:
                max_depths = [5, 10, None]
            else:
                max_depths = [max(current_max_depth - 3, 2), current_max_depth, current_max_depth + 3]
                
            current_n_estimators = current_params.get('n_estimators', 100)
            param_grid = {
                'max_depth': max_depths,
                'n_estimators': [
                    max(current_n_estimators - 50, 50), 
                    current_n_estimators, 
                    current_n_estimators + 50
                ],
                'min_samples_split': [5, 10, 15]
            }
            
        elif model_type == "GradientBoostingClassifier":
            # Parâmetros para GradientBoosting
            current_lr = current_params.get('learning_rate', 0.1)
            current_n_estimators = current_params.get('n_estimators', 100)
            current_max_depth = current_params.get('max_depth', 3)
            
            param_grid = {
                'learning_rate': [current_lr * 0.5, current_lr, current_lr * 1.5],
                'n_estimators': [
                    max(current_n_estimators - 50, 50), 
                    current_n_estimators, 
                    current_n_estimators + 50
                ],
                'max_depth': [
                    max(current_max_depth - 1, 1), 
                    current_max_depth, 
                    current_max_depth + 1
                ]
            }
        
        elif model_type == "XGBClassifier":
            # Parâmetros para XGBoost
            try:
                current_lr = current_params.get('learning_rate', 0.1)
                current_gamma = current_params.get('gamma', 0)
                
                param_grid = {
                    'learning_rate': [current_lr * 0.5, current_lr, current_lr * 1.5],
                    'gamma': [current_gamma, current_gamma + 0.1, current_gamma + 0.2],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9]
                }
            except Exception:
                # Se falhar na obtenção dos parâmetros, use valores padrão
                param_grid = {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'gamma': [0, 0.1, 0.2],
                    'subsample': [0.8],
                    'colsample_bytree': [0.8]
                }
        
        return param_grid