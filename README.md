# Predição de Churn de Clientes Bancários

## Visão Geral

Este projeto implementa um **modelo de machine learning para prever a saída (churn) de clientes de um banco**. O objetivo é identificar clientes com alto risco de deixar o banco para que ações de retenção possam ser implementadas proativamente.

**Dataset**: Customer-Churn-Records.csv (10.000 registros, 18 colunas)  
**Tipo de Problema**: Classificação Binária (Churn vs. Não-Churn)  
**Modelo Selecionado**: Rede Neural Artificial (TensorFlow/Keras)

---

## Resultados Alcançados

| Métrica | Rede Neural | Logistic Regression | Random Forest |
|---------|-----------|----------------------|---------------|
| **AUC-ROC** | **0.9986** | 0.9990 | 0.9990 |
| **F1-Score** | **0.9967** | - | - |
| **Acurácia** | 99.71% | - | - |
| **Sensibilidade (Recall)** | 99.67% | - | - |

A rede neural alcança performance excepcional, com praticamente nenhum erro em ambas as classes (apenas 4 erros em 3.000 amostras de teste: 2 falsos positivos e 2 falsos negativos).

---

## Estrutura do Projeto

```
├── 01-analise_exploratoria.ipynb          # Exploração inicial dos dados
├── 02-pre-processamento_dos_dados.ipynb   # Limpeza e transformação
├── 03-modelagem_preditiva.ipynb          # Treino de 10 modelos
├── modelo_neural_network.keras            # Modelo treinado (Keras)
├── scaler.pkl                             # StandardScaler serializado
├── requirements.txt                       # Dependências do projeto
├── README.md                              # Este arquivo
└── data/
    ├── Customer-Churn-Records.csv         # Dataset original
    ├── customer_churn_processed.csv       # Dataset processado
    ├── X_features.csv                     # Features (17 colunas)
    └── y_target.csv                       # Target (Exited)
```

---

## Dados

### Dataset Original
- **Tamanho**: 10.000 registros
- **Colunas**: 18 (incluindo ID, nome, dados demográficos, financeiros e comportamentais)
- **Target**: `Exited` (1 = churn, 0 = retém-se)
- **Desbalanceamento**: ~20% churn, ~80% retenção

### Pré-processamento Realizado

1. **Remoção de colunas irrelevantes**: RowNumber, CustomerId, Surname
2. **One-Hot Encoding**:
   - Geography (3 países): 2 dummies criadas
   - Gender (2 sexos): 1 dummy criada
   - Card Type (4 tipos): 3 dummies criadas
3. **Features Finais**: 17 features numéricas + 1 target
4. **Normalização**: StandardScaler aplicado ao conjunto de treino

### Features Utilizadas

| Feature | Descrição |
|---------|-----------|
| CreditScore | Pontuação de crédito (300-850) |
| Age | Idade do cliente (anos) |
| Tenure | Tempo como cliente (anos) |
| Balance | Saldo em conta (USD) |
| NumOfProducts | Número de produtos financeiros |
| HasCrCard | Possui cartão de crédito (0/1) |
| IsActiveMember | Membro ativo (0/1) |
| EstimatedSalary | Salário estimado (USD) |
| Complain | Reclamação anterior (0/1) |
| Satisfaction Score | Nível de satisfação (1-5) |
| Point Earned | Pontos acumulados |
| Geography_Germany | Localização = Alemanha (0/1) |
| Geography_Spain | Localização = Espanha (0/1) |
| Gender_Male | Gênero = Masculino (0/1) |
| Card Type_Gold | Tipo cartão = Ouro (0/1) |
| Card Type_Platinum | Tipo cartão = Platina (0/1) |
| Card Type_Silver | Tipo cartão = Prata (0/1) |

---

## Arquitetura do Modelo

### Modelo Selecionado: Rede Neural Artificial

```
Camada de Entrada: 17 features

Dense(64, relu) + BatchNormalization + Dropout(0.3)
    ↓
Dense(32, relu) + BatchNormalization + Dropout(0.2)
    ↓
Dense(16, relu)
    ↓
Dense(8, relu)
    ↓
Dense(1, sigmoid) → Probabilidade de Churn
```

### Hiperparâmetros

- **Otimizador**: Adam (learning_rate=0.0005)
- **Função de Perda**: Binary Crossentropy
- **Métricas**: Acurácia, AUC-ROC
- **Epochs**: 100 (com Early Stopping)
- **Batch Size**: 32
- **Validação**: 20% do conjunto de treino
- **Regularização**: L2 (0.001) + Dropout + BatchNormalization
- **Early Stopping**: Paciência=15 epochs, monitorando val_auc

### Justificativa de Escolha

Foram testados **10 modelos diferentes** durante a fase de modelagem:

- Logistic Regression (AUC=0.999)
- Decision Tree (AUC=0.994)
- Random Forest (AUC=0.999)
- Gradient Boosting (AUC=0.999)
- AdaBoost (AUC=0.998)
- XGBoost (AUC=0.998)
- LightGBM (AUC=0.998)
- SVM RBF (AUC=0.997)
- KNN (AUC=0.998)
- Naive Bayes (AUC=0.999)
- **Neural Network (AUC=0.9986)** ← Selecionado

A rede neural foi escolhida por apresentar o melhor equilíbrio entre performance (AUC muito próximo ao melhor), capacidade de captura de padrões complexos e potencial para generalização com dados novos.

---

## Como Usar

### 1. Instalação de Dependências

```bash
pip install -r requirements.txt
```

### 2. Treinar o Modelo

Execute os notebooks em sequência:

```bash
jupyter notebook 01-analise_exploratoria.ipynb
jupyter notebook 02-pre-processamento_dos_dados.ipynb
jupyter notebook 03-modelagem_preditiva.ipynb
```

### 3. Fazer Previsões

```python
import tensorflow as tf
import pickle
import pandas as pd

# Carregar modelo e scaler
modelo = tf.keras.models.load_model('modelo_neural_network.keras')
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Dados novo cliente (17 features)
novo_cliente = pd.DataFrame({...})  # 17 features

# Normalizar com o scaler de treino
novo_cliente_scaled = scaler.transform(novo_cliente)

# Fazer predição
probabilidade_churn = modelo.predict(novo_cliente_scaled)
classificacao = "Churn" if probabilidade_churn[0][0] > 0.5 else "Retém-se"
```

### 4. Avaliar Modelo

```python
# No notebook 03-modelagem_preditiva.ipynb

# Metricas no conjunto de teste
print(f"AUC-ROC: {auc_roc:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Matriz de Confusão:\n{cm}")

# Curva ROC
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC")
plt.show()
```

---

## Métricas e Interpretação

### Matriz de Confusão (Conjunto de Teste)

```
               Predito Não-Churn    Predito Churn
Atual Não-Churn    2.387                 2
Atual Churn            2                609
```

- **True Negatives (TN)**: 2.387 clientes corretamente identificados como não-churn
- **False Positives (FP)**: 2 clientes incorretamente preditos como churn
- **False Negatives (FN)**: 2 clientes que fizeram churn mas foram preditos como não-churn
- **True Positives (TP)**: 609 clientes corretamente identificados como churn

### Interpretação das Métricas

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| **Acurácia** | 99.71% | Das 3.000 previsões, 99.71% estavam corretas |
| **Precisão (Churn)** | 99.67% | De 611 preditos como churn, 609 realmente fizeram churn |
| **Recall (Sensibilidade)** | 99.67% | De 611 que realmente fizeram churn, 609 foram identificados |
| **F1-Score** | 0.9967 | Equilíbrio excelente entre precisão e recall |
| **AUC-ROC** | 0.9986 | O modelo diferencia muito bem entre classes (1.0 = perfeito) |

---

## Curva de Aprendizado

Durante o treino, a rede neural mostrou:

- **Epoch 1**: Val Loss=0.4571, Val AUC=0.9941
- **Epoch 2**: Val Loss=0.0959, Val AUC=0.9987
- **Epoch 3**: Val Loss=0.0467, Val AUC=0.9990
- **Convergência**: ~18 epochs (early stopping evitou overfitting)

A rápida convergência indica um problema bem-definido e dados de alta qualidade.

---

## Arquivos de Artefatos

### modelo_neural_network.keras

Arquivo serializado do modelo treinado em formato Keras (formato padrão do TensorFlow). Tamanho aproximado: 50 KB.

**Como carregar:**
```python
import tensorflow as tf
model = tf.keras.models.load_model('modelo_neural_network.keras')
```

### scaler.pkl

StandardScaler serializado via pickle. Contém a média e desvio-padrão de cada feature aprendidos no conjunto de treino.

**Por que salvar:**
- Garante que novos dados sejam normalizados exatamente como o treino
- Essencial para reproducibilidade em produção
- Evita data leakage

**Como carregar:**
```python
import pickle
scaler = pickle.load(open('scaler.pkl', 'rb'))
X_novo_scaled = scaler.transform(X_novo)
```

---

## Dependências

| Pacote | Versão | Propósito |
|--------|--------|----------|
| pandas | 1.5+ | Manipulação de dados |
| numpy | 1.20+ | Operações numéricas |
| scikit-learn | 1.0+ | Pré-processamento, métricas |
| tensorflow | 2.11+ | Rede neural, Keras |
| matplotlib | 3.5+ | Visualizações |
| seaborn | 0.12+ | Visualizações avançadas |

---

## Reprodutibilidade

### Seeds Fixadas

Para garantir reproducibilidade, os seeds foram fixados em:

- `random_state=42` (sklearn)
- `random_seed=42` (XGBoost, LightGBM)
- `seed=42` (TensorFlow - em desenvolvimento)

```python
# No notebook, no início:
import tensorflow as tf
tf.random.set_seed(42)
import numpy as np
np.random.seed(42)
import random
random.seed(42)
```

### Versões Exatas

Ver `requirements.txt` para versões exatas de todas as dependências.

---

## Licença

Este projeto é para fins educacionais.

---

**Última Atualização**: Novembro de 2025  
