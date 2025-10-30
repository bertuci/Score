"""
Correções para erros de Unicode logging e pickle serialization
"""

import logging
import sys
import io
import numpy as np
import pandas as pd

# =============================================================================
# FIX 1: Configurar logging para UTF-8 (compatível com Windows)
# =============================================================================

def setup_utf8_logging():
    """
    Configura o logging para usar UTF-8, resolvendo problemas com emojis no Windows
    """
    # Remove todos os handlers existentes
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Cria um handler com encoding UTF-8
    # No Windows, isso força o uso de UTF-8 em vez de cp1252
    if sys.platform == 'win32':
        # Para Windows, usar sys.stdout com UTF-8
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

    # Cria handler com UTF-8
    handler = logging.StreamHandler(
        stream=io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    )

    # Define o formato
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    # Configura o logger
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger


# =============================================================================
# FIX 2: Limpar pd.NA dos dados antes de operações problemáticas
# =============================================================================

def clean_pandas_na(df):
    """
    Substitui pd.NA por np.nan para evitar erros em operações sklearn/imblearn

    Args:
        df: DataFrame pandas

    Returns:
        DataFrame com pd.NA substituído por np.nan
    """
    # Criar uma cópia para não modificar o original
    df_clean = df.copy()

    # Substituir pd.NA por np.nan em todas as colunas
    for col in df_clean.columns:
        # Detectar se há pd.NA na coluna
        if df_clean[col].dtype == 'object' or pd.api.types.is_string_dtype(df_clean[col]):
            # Para colunas object/string, substituir pd.NA
            df_clean[col] = df_clean[col].fillna(np.nan)
        elif hasattr(pd, 'NA'):
            # Para colunas numéricas com pd.NA
            mask = df_clean[col].isna()
            if mask.any():
                df_clean.loc[mask, col] = np.nan

    # Converter tipos problemáticos
    for col in df_clean.columns:
        if pd.api.types.is_integer_dtype(df_clean[col]) and df_clean[col].isna().any():
            # Converter Int64/Int32 para float64 se tiver NaN
            df_clean[col] = df_clean[col].astype('float64')

    return df_clean


def ensure_numpy_array(X, y=None):
    """
    Garante que X e y são arrays numpy sem pd.NA

    Args:
        X: Features (DataFrame ou array)
        y: Target (Series ou array), opcional

    Returns:
        X_array, y_array (ou apenas X_array se y=None)
    """
    # Limpar X
    if isinstance(X, pd.DataFrame):
        X = clean_pandas_na(X)
        X_array = X.values.astype(np.float64)
    else:
        X_array = np.array(X, dtype=np.float64)

    # Substituir infinitos e NaN por valores numéricos
    X_array = np.nan_to_num(X_array, nan=0.0, posinf=1e10, neginf=-1e10)

    if y is not None:
        # Limpar y
        if isinstance(y, pd.Series):
            y = y.fillna(0)  # Para target, usar 0 como padrão
            y_array = y.values
        else:
            y_array = np.array(y)

        # Garantir que y não tem NaN
        if np.isnan(y_array).any():
            y_array = np.nan_to_num(y_array, nan=0)

        return X_array, y_array

    return X_array


# =============================================================================
# FIX 3: Wrapper para SMOTE que trata pd.NA
# =============================================================================

def safe_smote_fit_resample(smote_instance, X, y):
    """
    Wrapper seguro para SMOTE.fit_resample que trata pd.NA

    Args:
        smote_instance: Instância do SMOTE
        X: Features
        y: Target

    Returns:
        X_resampled, y_resampled
    """
    try:
        # Limpar os dados antes do SMOTE
        X_clean, y_clean = ensure_numpy_array(X, y)

        # Aplicar SMOTE
        X_resampled, y_resampled = smote_instance.fit_resample(X_clean, y_clean)

        return X_resampled, y_resampled

    except Exception as e:
        print(f"Erro no SMOTE: {e}")
        print("Retornando dados originais sem balanceamento")
        return ensure_numpy_array(X, y)


# =============================================================================
# FIX 4: Configuração completa para aplicar no notebook
# =============================================================================

def apply_all_fixes():
    """
    Aplica todas as correções de uma vez
    """
    print("Aplicando correções...")
    print("1. Configurando logging UTF-8...")
    logger = setup_utf8_logging()
    logger.info("✅ Logging UTF-8 configurado com sucesso!")

    print("2. Funções de limpeza de dados carregadas")
    print("3. Wrapper SMOTE seguro carregado")
    print()
    print("Correções aplicadas! Use as seguintes funções:")
    print("  - clean_pandas_na(df): Limpa pd.NA de um DataFrame")
    print("  - ensure_numpy_array(X, y): Converte para numpy array seguro")
    print("  - safe_smote_fit_resample(smote, X, y): SMOTE com tratamento de NA")

    return logger


if __name__ == "__main__":
    # Teste básico
    logger = apply_all_fixes()
    logger.info("🎉 Sistema pronto para uso!")
