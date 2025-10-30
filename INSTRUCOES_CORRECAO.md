# Instruções para Corrigir Erros de Unicode e Pickle

## Problemas Identificados

1. **UnicodeEncodeError**: Emojis no logging não funcionam no console Windows (cp1252)
2. **TypeError (pickle/NA)**: `pd.NA` causa erro "boolean value of NA is ambiguous" no SMOTE

## Solução Rápida

### Opção 1: Adicionar no Início do Notebook

Adicione esta célula **NO INÍCIO** do seu notebook (antes de qualquer código):

```python
# ============================================================================
# CORREÇÃO: Unicode Logging + pd.NA Issues
# ============================================================================

import logging
import sys
import io
import numpy as np
import pandas as pd

# FIX 1: Configurar logging UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Remover handlers existentes
logger = logging.getLogger()
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Criar handler com UTF-8
handler = logging.StreamHandler(
    stream=io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
)
formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

print("✅ Logging UTF-8 configurado!")
```

### Opção 2: Modificar a Classe de Balanceamento

Encontre a classe `SmartBalancer` e modifique o método `fit_resample`:

**ANTES:**
```python
def fit_resample(self, X, y):
    try:
        X_resampled, y_resampled = self.sampler.fit_resample(X, y)
        # ...
```

**DEPOIS:**
```python
def fit_resample(self, X, y):
    try:
        # Converter para numpy e limpar pd.NA
        if isinstance(X, pd.DataFrame):
            X = X.fillna(0).values.astype(np.float64)
        else:
            X = np.array(X, dtype=np.float64)

        if isinstance(y, pd.Series):
            y = y.fillna(0).values
        else:
            y = np.array(y)

        # Limpar NaN e infinitos
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        y = np.nan_to_num(y, nan=0)

        X_resampled, y_resampled = self.sampler.fit_resample(X, y)
        # ...
```

### Opção 3: Usar o Script de Correção

Execute no início do notebook:

```python
# Carregar correções
exec(open('fix_unicode_and_pickle_errors.py').read())

# Aplicar todas as correções
logger = apply_all_fixes()

# Agora use clean_pandas_na() antes de operações problemáticas
# Exemplo:
# df_clean = clean_pandas_na(df)
# X_clean, y_clean = ensure_numpy_array(X, y)
```

## Correção Específica para cada Erro

### Para UnicodeEncodeError

**No Windows PowerShell/CMD:**
```powershell
# Antes de executar Python
chcp 65001
python seu_script.py
```

**Ou configurar variável de ambiente:**
```powershell
$env:PYTHONIOENCODING="utf-8"
python seu_script.py
```

**No Jupyter Notebook:**
Adicione no início:
```python
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
```

### Para TypeError: boolean value of NA is ambiguous

**Antes de qualquer operação com SMOTE/sklearn:**
```python
# Limpar pd.NA de DataFrames
def clean_na(df):
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(0)
    return df

# Usar antes do SMOTE
X_train = clean_na(X_train)
```

## Correção Permanente no Notebook

Localize a célula que define `logger` e substitua por:

```python
import logging
import sys
import io

# Configuração UTF-8 para Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# Criar logger com UTF-8
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Remover handlers antigos
for h in logger.handlers[:]:
    logger.removeHandler(h)

# Handler com UTF-8
try:
    handler = logging.StreamHandler(
        stream=io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    )
except:
    handler = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
```

## Teste Rápido

Execute para verificar se funcionou:

```python
import logging
logger = logging.getLogger()
logger.info("🚀 Teste de emoji")
logger.info("✅ Se você vê isso, está funcionando!")
```

Se ainda houver erro, tente:

```python
# Remover todos os emojis do código (buscar e substituir)
# 🚀 → [START]
# ✅ → [OK]
# ⚠️ → [WARN]
# ❌ → [ERROR]
# etc.
```

## Alternativa: Executar sem Emojis

Se preferir remover os emojis do código:

```python
# Função para substituir emojis
def remove_emojis_from_logs():
    import re
    # Criar um handler que remove emojis
    class NoEmojiHandler(logging.StreamHandler):
        def emit(self, record):
            record.msg = re.sub(r'[^\x00-\x7F]+', '', str(record.msg))
            super().emit(record)

    logger = logging.getLogger()
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    logger.addHandler(NoEmojiHandler())

remove_emojis_from_logs()
```

## Resumo das Mudanças Necessárias

1. ✅ Configurar UTF-8 no início do notebook
2. ✅ Limpar pd.NA antes do SMOTE
3. ✅ Converter DataFrames para numpy arrays antes de sklearn operations
4. ✅ Usar fillna() ou np.nan_to_num() para tratar NaN

Execute estas correções e teste novamente!
