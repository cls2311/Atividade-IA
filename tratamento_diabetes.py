import pandas as pd
import numpy as np
import re
from datetime import datetime

# ----------------------------------------------------------------------
# 1. Carregar o conjunto de dados
# ----------------------------------------------------------------------
df = pd.read_csv('diabetes_prova.csv', encoding='utf-8')
print("Formato inicial:", df.shape)
print("\nNomes das colunas originais:", df.columns.tolist())
print("\nPrimeiras 5 linhas:\n", df.head())

# ----------------------------------------------------------------------
# 2. Padronizar nomes das colunas (remover acentos, espaços, caracteres especiais)
# ----------------------------------------------------------------------
def remove_accents(text):
    mapping = {
        'á': 'a', 'à': 'a', 'ã': 'a', 'â': 'a',
        'é': 'e', 'ê': 'e',
        'í': 'i', 'î': 'i',
        'ó': 'o', 'ô': 'o', 'õ': 'o',
        'ú': 'u', 'û': 'u',
        'ç': 'c',
        ' ': '_'
    }
    for k, v in mapping.items():
        text = text.replace(k, v)
    text = re.sub(r'[^a-zA-Z0-9_]', '', text)
    return text

df.columns = [remove_accents(col.lower()) for col in df.columns]

# Renomear colunas para simplificar
df.rename(columns={
    'glicose_jejum': 'glicose',
    'data_do_ultimo_exame_de_sangue': 'data_exame'
}, inplace=True)

print("\nColunas padronizadas:", df.columns.tolist())

# ----------------------------------------------------------------------
# 3. Limpar e padronizar 'sexo' (categórica)
# ----------------------------------------------------------------------
sexo_map = {
    'M': 'M', 'Masculino': 'M', 'Homem': 'M', 'Menino': 'M',
    'F': 'F', 'Feminino': 'F', 'Mulher': 'F', 'Menina': 'F',
    'FF': 'F', 'MM': 'M'  # correções de digitação
}
df['sexo'] = df['sexo'].map(sexo_map).fillna(np.nan)

# Imputar pela moda
moda_sexo = df['sexo'].mode()[0] if not df['sexo'].mode().empty else 'M'
df['sexo'] = df['sexo'].fillna(moda_sexo)

# ----------------------------------------------------------------------
# 4. Limpar 'peso' (numérica)
# ----------------------------------------------------------------------
df['peso'] = pd.to_numeric(df['peso'], errors='coerce')
df.loc[df['peso'] <= 0, 'peso'] = np.nan          # peso zero ou negativo é inválido
mediana_peso = df['peso'].median()
df['peso'] = df['peso'].fillna(mediana_peso)

# ----------------------------------------------------------------------
# 5. Limpar 'glicose' com restrições [90, 180]
# ----------------------------------------------------------------------
df['glicose'] = pd.to_numeric(df['glicose'], errors='coerce')
df.loc[(df['glicose'] < 90) | (df['glicose'] > 180), 'glicose'] = np.nan
mediana_glicose = df['glicose'].median()
df['glicose'] = df['glicose'].fillna(mediana_glicose)

# ----------------------------------------------------------------------
# 6. Limpar 'pressao' com restrições [10, 18]
# ----------------------------------------------------------------------
df['pressao'] = pd.to_numeric(df['pressao'], errors='coerce')
df.loc[(df['pressao'] < 10) | (df['pressao'] > 18), 'pressao'] = np.nan
mediana_pressao = df['pressao'].median()
df['pressao'] = df['pressao'].fillna(mediana_pressao)

# ----------------------------------------------------------------------
# 7. Limpar 'insulina' com restrições [0, 4]
# ----------------------------------------------------------------------
df['insulina'] = pd.to_numeric(df['insulina'], errors='coerce')
df.loc[(df['insulina'] < 0) | (df['insulina'] > 4), 'insulina'] = np.nan
mediana_insulina = df['insulina'].median()
df['insulina'] = df['insulina'].fillna(mediana_insulina)

# ----------------------------------------------------------------------
# 8. Limpar 'idade' (numérica)
# ----------------------------------------------------------------------
df['idade'] = pd.to_numeric(df['idade'], errors='coerce')
df.loc[(df['idade'] < 0) | (df['idade'] > 120), 'idade'] = np.nan
mediana_idade = df['idade'].median()
df['idade'] = df['idade'].fillna(mediana_idade)

# ----------------------------------------------------------------------
# 9. Limpar 'data_exame'
# ----------------------------------------------------------------------
def parse_date(date_str):
    if pd.isna(date_str):
        return np.nan
    date_str = str(date_str).strip()
    try:
        return pd.to_datetime(date_str, dayfirst=True, errors='coerce')
    except:
        pass
    months_pt = {
        'janeiro': 1, 'fevereiro': 2, 'março': 3, 'abril': 4, 'maio': 5, 'junho': 6,
        'julho': 7, 'agosto': 8, 'setembro': 9, 'outubro': 10, 'novembro': 11, 'dezembro': 12
    }
    match = re.search(r'(\d{1,2})\s+de\s+([a-zç]+)\s+de\s+(\d{4})', date_str.lower())
    if match:
        day = int(match.group(1))
        month_name = match.group(2)
        year = int(match.group(3))
        month = months_pt.get(month_name)
        if month:
            try:
                return datetime(year, month, day)
            except ValueError:
                return np.nan
    return np.nan

df['data_exame'] = df['data_exame'].apply(parse_date)
df['data_exame'] = pd.to_datetime(df['data_exame'], errors='coerce')
df = df.dropna(subset=['data_exame'])            # remove registros sem data válida
df['data_exame'] = df['data_exame'].dt.strftime('%d/%m/%Y')

# ----------------------------------------------------------------------
# 10. Remover duplicatas baseadas no ID do paciente
# ----------------------------------------------------------------------
# A primeira coluna é o identificador. Vamos renomeá-la para 'paciente_id'
if 'paciente_id' not in df.columns:
    # O nome original pode ter sido transformado para 'paciente_id' ou 'paciente_id'
    if 'paciente_id' in df.columns:
        pass
    else:
        df.rename(columns={df.columns[0]: 'paciente_id'}, inplace=True)

df = df.drop_duplicates(subset=['paciente_id'], keep='first')

# ----------------------------------------------------------------------
# 11. Verificação final e salvamento
# ----------------------------------------------------------------------
df.reset_index(drop=True, inplace=True)

print("\nFormato após limpeza:", df.shape)
print("\nValores ausentes por coluna:\n", df.isnull().sum())

df.to_csv('diabetes_limpo.csv', index=False, encoding='utf-8-sig')
print("\nDados limpos salvos em 'diabetes_limpo.csv'")