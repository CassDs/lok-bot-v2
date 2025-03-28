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
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
        indicator_types = ["SMA", "EMA", "RSI", "MACD", "Bollinger Bands"]
        ttk.Combobox(row1, textvariable=self.indicator_type_var, 
                    values=indicator_types, width=15, state="readonly").pack(side=tk.LEFT, padx=5)
        
        ttk.Label(row1, text="Período:").pack(side=tk.LEFT, padx=(20,5))
        self.indicator_period_var = StringVar(value="14")
        ttk.Entry(row1, textvariable=self.indicator_period_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # Botões para adicionar/remover indicadores
        row2 = ttk.Frame(indicators_frame)
        row2.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(row2, text="Adicionar Indicador", 
                  command=self._add_technical_indicator).pack(side=tk.LEFT, padx=5)
        
        # Lista de indicadores adicionados
        row3 = ttk.Frame(indicators_frame)
        row3.pack(fill="x", padx=5, pady=5)
        
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
        ttk.Radiobutton(split_method_frame, text="Divisão temporal (recomendado para dados financeiros)", 
                       variable=self.split_method_var, value="time").pack(anchor="w", padx=5, pady=2)
        
        ttk.Radiobutton(split_method_frame, text="Divisão aleatória", 
                       variable=self.split_method_var, value="random").pack(anchor="w", padx=5, pady=2)
        
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
            "GradientBoosting - Alta precisão, mais lento"
        ]
        
        ttk.Combobox(model_row, textvariable=self.algorithm_var, 
                   values=algorithms, width=40, state="readonly").pack(side=tk.LEFT, padx=5)
        
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
        
        # Frame para o gráfico
        self.plot_frame = ttk.Frame(viz_frame)
        self.plot_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Botões para salvar/carregar o modelo
        save_frame = ttk.Frame(parent)
        save_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(save_frame, text="Salvar Modelo", 
                  command=self._salvar_modelo).pack(side=tk.LEFT, padx=5)
                  
        ttk.Button(save_frame, text="Carregar Modelo", 
                  command=self._carregar_modelo).pack(side=tk.LEFT, padx=5)
    
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
            
        indicator_type = self.indicator_type_var.get()
        
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
            else:
                # Divisão aleatória
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, train_size=train_size, random_state=42
                )
            
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
            
            # Criar e treinar o modelo
            algorithm = self.algorithm_var.get().split(" - ")[0]  # Remover descrição
            
            if algorithm == "LogisticRegression":
                model = LogisticRegression(max_iter=1000, random_state=42)
            elif algorithm == "RandomForest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif algorithm == "GradientBoosting":
                from sklearn.ensemble import GradientBoostingClassifier
                model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Treinar o modelo
            model.fit(X_train, y_train)
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
                "O modelo está pronto para fazer previsões!"
            )
            
            # Atualizar interface no thread principal
            self.frame.after(0, lambda: self._atualizar_apos_treino(results, test_acc))
                
        except Exception as e:
            # Substitua self.master.after por self.frame.after
            self.frame.after(0, lambda: self._erro_treino(str(e)))
            import traceback
            traceback.print_exc()
    
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
        self.model_status_var.set(f"Modelo: {self.algorithm_var.get().split(' - ')[0]}")
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
                'date_trained': datetime.now().strftime('%Y-%m-%d %H:%M')
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
            
            # Atualizar interface
            self.model_status_var.set(f"Modelo: {algorithm} (carregado de arquivo)")
            self.model_accuracy_var.set(f"Acurácia: {accuracy:.4f} ({accuracy*100:.1f}%)")
            
            # Mostrar mensagem nos resultados
            results = (
                f"Modelo Carregado: {algorithm}\n"
                f"Data de treinamento: {date_trained}\n"
                f"Acurácia reportada: {accuracy:.4f} ({accuracy*100:.1f}%)\n\n"
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
                X_recent = self.scaler.transform(X_recent)
                
            # Fazer predição
            predictions = self.model.predict(X_recent)
            
            # Tentar obter probabilidades
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                try:
                    probabilities = self.model.predict_proba(X_recent)
                except:
                    pass
                    
            # Montar resultados para exibição
            result_text = "=== Previsão em Dados Recentes ===\n\n"
            result_text += f"Usando {num_candles} candles dos dados recentes\n\n"
            
            # Incluir timestamps se disponíveis
            has_timestamp = False
            timestamps = []
            
            if 'timestamp' in recent_data.columns:
                timestamps = recent_data['timestamp'].tail(num_candles).tolist()
                has_timestamp = True
                result_text += f"{'Data/Hora':<20} {'Previsão':<10} {'Confiança':<10}\n"
            else:
                result_text += f"{'#':<5} {'Previsão':<10} {'Confiança':<10}\n"
                
            result_text += "-" * 40 + "\n"
            
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
                result_text += f"{confidence:<10}\n"
                
            # Mostrar resultados
            self.prediction_text.config(state=tk.NORMAL)
            self.prediction_text.delete('1.0', tk.END)
            self.prediction_text.insert(tk.END, result_text)
            self.prediction_text.config(state=tk.DISABLED)
            
            # Plotar gráfico dos dados recentes com previsão
            self._plot_recent_prediction(recent_data.tail(num_candles), predictions)
            
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
        
    def _plot_recent_prediction(self, recent_data, predictions):
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