
## MVP – Ciência de Dados: Análise Preditiva para Violação de Dados - Uma abordagem comparativa entre Modelos

**Discente:** *\[Evanei Gomes dos Santos]*

**Docente:** *\[Andre Luiz Marques Serrano]*

**Data:** *\[28/09/2025]*


---

## 1. Definição do Problema

Organizações públicas e privadas enfrentam um aumento de **violações de dados**, expondo informações sensíveis e causando prejuízos financeiros e danos à reputação. O custo médio de uma violação de dados, segundo a IBM Security, é de **US$ 4,88 milhões**. Esses incidentes não são aleatórios, mas seguem **padrões temporais e setoriais** que, se compreendidos, podem ajudar a antecipar picos e otimizar a prevenção e a resposta.

Este MVP busca prever a quantidade mensal de violações de dados por tipo de organização (ex.: saúde, varejo, governo), a partir de séries temporais históricas (2010–2023) extraídas do dataset Privacy Rights Clearinghouse – Data Breach Chronology

* **Prophet (Meta/Facebook)** – modelo estatístico aditivo, robusto para sazonalidade.
* **ARIMA/SARIMA** – modelo estatístico clássico para séries temporais.
* **XGBoost Regressor** – modelo de aprendizado de máquina baseado em árvores de decisão (Gradient Boosting).

O estudo atende aos objetivos do **MVP/Prova 1 do Programa de Pós-Graduação Profissional em Engenharia Elétrica (PPEE/UnB)** e faz parte de uma linha de pesquisa sobre **cibersegurança e predição de incidentes**, fundamentando-se em **modelos comparativos** para auxiliar **estratégias de mitigação de riscos e políticas de segurança da informação**.

Hipótese

As tendências históricas de violações de dados, agregadas mensalmente por setor organizacional, contêm informações suficientes para gerar previsões confiáveis sobre incidentes futuros.
O uso de modelos de séries temporais e de aprendizado de máquina permite reduzir o erro de previsão (MAPE) para níveis aceitáveis (< 20%) em setores com maior regularidade histórica.

---

## 2. Dataset

**Fonte:** [Privacy Rights Clearinghouse – Data Breach Chronology](https://privacyrights.org)
**Período:** **2010 – 2023** (registros anteriores a 2010 foram desconsiderados por baixa consistência)
**Periodicidade:** agregação **mensal (ME)** para as séries temporais.

**Atributos Principais:**

| Coluna        | Descrição                                                     |
| ------------- | ------------------------------------------------------------- |
| `Date Breach` | Data do incidente                                             |
| `BSF`         | Serviços financeiros                                          |
| `BSO`         | Outros negócios (TI, manufatura, serviços)                    |
| `BSR`         | Varejo (lojas físicas e online)                               |
| `EDU`         | Educação (escolas, universidades)                             |
| `GOV`         | Governo e forças armadas                                      |
| `MED`         | Saúde (hospitais, clínicas)                                   |
| `NGO`         | Organizações sem fins lucrativos                              |
| `UNKN`        | Setor desconhecido (não classificado por falta de informação) |
| `Total Geral` | Soma de todos os setores                                      |

**Pré-Processamento:**

* Ajuste de datas incompletas: descartadas datas apenas com ano (`YYYY`), e datas `YYYY-MM` assumiram dia 1.
* Filtro temporal aplicado: `2010-01-01` a `2023-12-31`.
* Reamostragem mensal: `df.resample('ME').sum()`.
* Tratamento de **outliers** via **IQR (Interquartile Range)** para maior robustez dos modelos.
* Cálculo do **Expoente de Hurst** para avaliar persistência ou aleatoriedade das séries.

---

### Configuração do Ambiente (Python/Colab)

```python
# Bibliotecas principais
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Modelos e métricas
from prophet import Prophet                # Modelo Prophet (Meta/Facebook)
from statsmodels.tsa.statespace.sarimax import SARIMAX  # ARIMA/SARIMA
from xgboost import XGBRegressor           # Modelo XGBoost
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configuração de visualização
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("✅ Ambiente configurado e bibliotecas importadas com sucesso!")
```


---

## 3. Ambiente e Dependências

**Linguagem:** Python **3.10+**
**IDE recomendada:** [Google Colab](https://colab.research.google.com/) – para execução interativa em nuvem.

### Instalação das Dependências

```bash
pip install prophet statsmodels xgboost scikit-learn pandas numpy matplotlib seaborn openpyxl
```

### Principais Bibliotecas Utilizadas

* **prophet** – modelo estatístico aditivo para séries temporais.
* **statsmodels** – implementação do **ARIMA/SARIMA**.
* **xgboost** – regressão baseada em *gradient boosting* para padrões não lineares.
* **scikit-learn** – métricas (MAE, RMSE, MAPE) e funções auxiliares.
* **pandas / numpy** – ETL e manipulação numérica.
* **matplotlib / seaborn** – visualizações gráficas e heatmaps.
* **openpyxl** – leitura e escrita de arquivos Excel.

### Configuração de Estilo dos Gráficos

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Estilo visual
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

print("✅ Ambiente configurado com sucesso!")
```

> 💡 O ambiente foi validado no **Google Colab** e é compatível com ambientes locais que utilizem Python 3.10 ou superior.


## 4. Preparação dos Dados

A preparação dos dados teve como objetivo **organizar as séries temporais** para os modelos Prophet, ARIMA e XGBoost, garantindo consistência entre setores e períodos.

As etapas realizadas foram:

1. **Carregamento do Dataset**

   * O notebook lê a base consolidada de violações de dados por setor, previamente salva em planilha.

2. **Ajuste de Datas**

   * A coluna com as datas dos incidentes foi padronizada para um único formato válido.
   * Foram descartados registros com datas incompletas (por exemplo, apenas ano).

3. **Filtro Temporal**

   * O conjunto foi restrito ao período de **janeiro/2010 a dezembro/2023**.

4. **Agregação Mensal**

   * As séries foram **reamostradas em frequência mensal (ME)** para consolidar a contagem de incidentes por mês e por setor.

5. **Análise do Expoente de Hurst**

   * Foi calculado o **expoente de Hurst** para cada setor, a fim de verificar se as séries apresentavam **persistência, reversão à média ou comportamento aleatório**.

---

## 5. Métricas de Avaliação

O notebook realiza a **avaliação comparativa do desempenho dos modelos Prophet, ARIMA e XGBoost** na previsão do número mensal de violações de dados por setor organizacional.

### Métricas de Avaliação

Foram utilizadas métricas padrão para séries temporais:

* **MAE (Mean Absolute Error)** – erro médio absoluto.
* **RMSE (Root Mean Square Error)** – raiz do erro quadrático médio.
* **MAPE (%) (Mean Absolute Percentage Error)** – erro percentual absoluto médio.

> 🔎 O **MAPE (%)** é destacado como **métrica principal** por permitir **comparação proporcional entre setores com diferentes magnitudes**.

---

## 6. Resultados Principais

O desempenho foi avaliado por **MAPE (%)** para cada setor e modelo.

> 🔎 Observações:
>
> * **XGBoost** apresentou **menores MAPE em setores como Total Geral (5,97%) e UNKN (10,03%)**, obtendo **boa/alta precisão**.
> * **Prophet** teve performance intermediária, com destaque para **MED (26,54%)** e **Total Geral (17,63%)** (boa precisão).
> * **ARIMA** obteve MAPE mais altos, sendo competitivo apenas em **EDU (36,74%)**.
> * O **setor BSR (Varejo)** foi o mais desafiador (MAPE ≥ 70% em todos os modelos – previsão imprecisa).

---

## 7. Conclusão

A análise evidencia que **a métrica MAPE foi determinante para identificar a acurácia relativa entre setores e modelos**, validando achados do artigo de referência.

* O **XGBoost** se destacou por **menor erro percentual em setores agregados (Total Geral e UNKN)**, tornando-se mais indicado para séries com maior volume e padrões suaves.
* O **Prophet** apresentou **desempenho consistente em setores com sazonalidade clara (MED, EDU)**.
* O **ARIMA** mostrou-se mais limitado em cenários complexos ou de alta variabilidade.
* O **MAPE elevado no BSR** sugere **alta volatilidade no varejo**, exigindo abordagens mais sofisticadas (ex.: redes neurais LSTM/TCN, conforme evidências do artigo original).

Esses resultados reforçam a importância da **seleção contextual do modelo** e demonstram como métricas interpretáveis, como **MAPE**, orientam a **tomada de decisão em políticas de segurança cibernética**.

> **Perspectivas Futuras:** incorporar **modelos de deep learning (LSTM, TCN)**, explorar **previsão por tipo de vazamento** e **expandir a análise para bases globais**, alinhando-se às recomendações de trabalhos recentes.

---

## 8. Estrutura do Repositório

```
├── OrganizationType_Prophet_x_Arima_x_Xgboost_v2.ipynb
├── resultados_prophet_gridsearch.csv
├── resultados_arima_gridsearch.csv
├── melhores_resultados_xgboost.csv
├── imagens/
│   └── heatmap_mape.jpg
└── README.md
```

---

**Autor:** Evanei Gomes dos Santos – PPEE/UnB 2025
[evanei.santos@aluno.unb.br](mailto:evanei.santos@aluno.unb.br)

---

