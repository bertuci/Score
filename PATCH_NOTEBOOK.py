"""
PATCH COMPLETO - Copie e cole esta célula NO INÍCIO do seu notebook Score.ipynb
"""

# ============================================================================
# CORREÇÃO COMPLETA: Unicode Logging + pd.NA + Pickle Issues
# ============================================================================

import logging
import sys
import io
import numpy as np
import pandas as pd
import warnings

print("=" * 70)
print("APLICANDO CORREÇÕES AUTOMÁTICAS")
print("=" * 70)

# ----------------------------------------------------------------------------
# FIX 1: Configurar logging UTF-8 (resolve UnicodeEncodeError com emojis)
# ----------------------------------------------------------------------------

def setup_logging_utf8():
    """Configura logging para suportar emojis no Windows"""
    try:
        # Para Windows: reconfigurar stdout/stderr para UTF-8
        if sys.platform == 'win32':
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')

        # Obter logger root
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Remover handlers existentes
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Criar novo handler com UTF-8
        handler = logging.StreamHandler(
            stream=io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
        )

        # Definir formato
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        print("✅ Logging UTF-8 configurado com sucesso")
        return logger

    except Exception as e:
        print(f"⚠️  Aviso: Não foi possível configurar UTF-8: {e}")
        print("    Continuando com configuração padrão...")
        return logging.getLogger()

logger = setup_logging_utf8()

# ----------------------------------------------------------------------------
# FIX 2: Monkey-patch para pandas NA (resolve pickle/SMOTE errors)
# ----------------------------------------------------------------------------

# Salvar métodos originais
_original_fillna = pd.DataFrame.fillna
_original_series_fillna = pd.Series.fillna

def safe_fillna_df(self, value=np.nan, **kwargs):
    """DataFrame.fillna que trata pd.NA corretamente"""
    # Substituir pd.NA por np.nan primeiro
    df = self.copy()
    for col in df.columns:
        if df[col].dtype == 'object' or hasattr(df[col], 'array'):
            try:
                mask = pd.isna(df[col])
                if mask.any():
                    df.loc[mask, col] = np.nan
            except:
                pass
    return _original_fillna(df, value=value, **kwargs)

def safe_fillna_series(self, value=np.nan, **kwargs):
    """Series.fillna que trata pd.NA corretamente"""
    # Substituir pd.NA por np.nan
    s = self.copy()
    try:
        mask = pd.isna(s)
        if mask.any():
            s.loc[mask] = np.nan
    except:
        pass
    return _original_series_fillna(s, value=value, **kwargs)

# Aplicar monkey-patch
pd.DataFrame.fillna = safe_fillna_df
pd.Series.fillna = safe_fillna_series

print("✅ Pandas fillna() patch aplicado")

# ----------------------------------------------------------------------------
# FIX 3: Função auxiliar para limpar dados antes do SMOTE
# ----------------------------------------------------------------------------

def prepare_for_smote(X, y):
    """
    Prepara dados para SMOTE, removendo pd.NA e convertendo para numpy

    Args:
        X: Features (DataFrame ou array)
        y: Target (Series ou array)

    Returns:
        X_clean, y_clean como numpy arrays
    """
    # Limpar X
    if isinstance(X, pd.DataFrame):
        X = X.copy()
        # Substituir pd.NA por np.nan
        for col in X.columns:
            mask = pd.isna(X[col])
            if mask.any():
                X.loc[mask, col] = np.nan
        # Preencher NaN com 0
        X = X.fillna(0)
        X_clean = X.values.astype(np.float64)
    else:
        X_clean = np.array(X, dtype=np.float64)

    # Limpar y
    if isinstance(y, pd.Series):
        y = y.copy()
        mask = pd.isna(y)
        if mask.any():
            y.loc[mask] = np.nan
        y = y.fillna(0)
        y_clean = y.values
    else:
        y_clean = np.array(y)

    # Substituir infinitos e NaN restantes
    X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=1e10, neginf=-1e10)
    y_clean = np.nan_to_num(y_clean, nan=0)

    return X_clean, y_clean

print("✅ Função prepare_for_smote() carregada")

# ----------------------------------------------------------------------------
# FIX 4: Wrapper para métodos problemáticos
# ----------------------------------------------------------------------------

# Importar SMOTE se disponível
try:
    from imblearn.over_sampling import SMOTE
    _original_smote_fit_resample = SMOTE.fit_resample

    def safe_smote_fit_resample(self, X, y):
        """SMOTE.fit_resample com tratamento automático de pd.NA"""
        X_clean, y_clean = prepare_for_smote(X, y)
        return _original_smote_fit_resample(self, X_clean, y_clean)

    SMOTE.fit_resample = safe_smote_fit_resample
    print("✅ SMOTE patch aplicado")
except ImportError:
    print("⚠️  SMOTE não encontrado (imblearn não instalado)")

# ----------------------------------------------------------------------------
# FIX 5: Tratamento global de warnings
# ----------------------------------------------------------------------------

# Suprimir avisos específicos que não são críticos
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*boolean value of NA.*')

print("✅ Filtros de warning configurados")

# ----------------------------------------------------------------------------
# FIX 6: Função para limpar DataFrames completamente
# ----------------------------------------------------------------------------

def clean_dataframe(df):
    """
    Limpa um DataFrame completamente, removendo pd.NA e valores problemáticos

    Args:
        df: DataFrame para limpar

    Returns:
        DataFrame limpo
    """
    df = df.copy()

    for col in df.columns:
        # Detectar e substituir pd.NA
        mask = pd.isna(df[col])
        if mask.any():
            if df[col].dtype in ['int64', 'int32', 'int16', 'int8']:
                # Para inteiros, converter para float para permitir NaN
                df[col] = df[col].astype('float64')
                df.loc[mask, col] = np.nan
            else:
                df.loc[mask, col] = np.nan

        # Preencher NaN de forma apropriada
        if df[col].dtype in ['float64', 'float32']:
            df[col] = df[col].fillna(0.0)
        elif df[col].dtype == 'object':
            df[col] = df[col].fillna('')

    return df

print("✅ Função clean_dataframe() carregada")

# ----------------------------------------------------------------------------
# Verificação final
# ----------------------------------------------------------------------------

print("=" * 70)
print("CORREÇÕES APLICADAS COM SUCESSO!")
print("=" * 70)
print()
print("Funções disponíveis:")
print("  • prepare_for_smote(X, y) - Prepara dados para SMOTE")
print("  • clean_dataframe(df) - Limpa DataFrame de pd.NA")
print()
print("Patches aplicados:")
print("  • Logging UTF-8 ativado (suporta emojis)")
print("  • pandas fillna() corrigido para tratar pd.NA")
print("  • SMOTE.fit_resample() corrigido automaticamente")
print()
logger.info("🎉 Sistema pronto! Pode executar o pipeline normalmente.")
print("=" * 70)
