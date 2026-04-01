import pandas as pd
import numpy as np
import re
from datetime import datetime

# ----------------------------------------------------------------------
# 1. Carregar o conjunto de dados
# ----------------------------------------------------------------------
df = pd.read_csv('dados_covid_prova.csv', encoding='utf-8')
print("Formato inicial:", df.shape)
print("\nNomes das colunas:", df.columns.tolist())
print("\nPrimeiras 5 linhas:\n", df.head())

# ----------------------------------------------------------------------
# 2. Padronizar nomes das colunas (remover acentos e caracteres especiais)
# ----------------------------------------------------------------------
def remove_accents(text):
    # Mapeamento de caracteres acentuados para seus equivalentes sem acento
    mapping = {
        'á': 'a', 'à': 'a', 'ã': 'a', 'â': 'a',
        'é': 'e', 'ê': 'e',
        'í': 'i', 'î': 'i',
        'ó': 'o', 'ô': 'o', 'õ': 'o',
        'ú': 'u', 'û': 'u',
        'ç': 'c',
        ' ': '_'  # substitui espaços por underscore
    }
    for k, v in mapping.items():
        text = text.replace(k, v)
    # Remove qualquer outro caractere que não seja letra, número ou underscore
    text = re.sub(r'[^a-zA-Z0-9_]', '', text)
    return text

# Aplica a padronização a todos os nomes de coluna
new_columns = [remove_accents(col.lower()) for col in df.columns]
df.columns = new_columns

# Garante que a coluna de doenças tenha um nome simples e único
if 'doencas_respiratoria' in df.columns:
    df.rename(columns={'doencas_respiratoria': 'doenca_respiratoria'}, inplace=True)

print("\nNomes das colunas padronizados:", df.columns.tolist())

# ----------------------------------------------------------------------
# 3. Limpar e padronizar 'covid_positivo'
# ----------------------------------------------------------------------
df['covid_positivo'] = df['covid_positivo'].replace({'Ss': 'Sim', 'N': 'Não'}).fillna('Não')
valid_covid = ['Sim', 'Não']
df = df[df['covid_positivo'].isin(valid_covid)]

# ----------------------------------------------------------------------
# 4. Limpar 'genero'
# ----------------------------------------------------------------------
gender_map = {
    'Masculino': 'Masculino',
    'Menino': 'Masculino',
    'Homem': 'Masculino',
    'Feminino': 'Feminino',
    'Menina': 'Feminino',
    'Mulher': 'Feminino',
    'SN': np.nan
}
df['genero'] = df['genero'].map(gender_map).fillna(np.nan)

# ----------------------------------------------------------------------
# 5. Limpar 'idade'
# ----------------------------------------------------------------------
df['idade'] = pd.to_numeric(df['idade'], errors='coerce')
df.loc[df['idade'] > 120, 'idade'] = np.nan
df.loc[df['idade'] < 0, 'idade'] = np.nan
median_age = df['idade'].median()
df['idade'] = df['idade'].fillna(median_age)

# ----------------------------------------------------------------------
# 6. Limpar 'febre'
# ----------------------------------------------------------------------
df['febre'] = pd.to_numeric(df['febre'], errors='coerce')
df.loc[(df['febre'] < 36) | (df['febre'] > 40), 'febre'] = np.nan

# ----------------------------------------------------------------------
# 7. Limpar 'Cidade'
# ----------------------------------------------------------------------
city_map = {
    'João Pessoa': 'João Pessoa',
    'JP': 'João Pessoa',
    'Campina Grande': 'Campina Grande',
    'CG': 'Campina Grande',
    'Patos': 'Patos',
    'Cajazeiras': 'Cajazeiras',
    'Cabedelo': 'Cabedelo',
    'Sousa': 'Sousa',
    'Souza': 'Sousa'
}
# Aplica o mapeamento (mantém o nome original com acentos para as cidades)
df['cidade'] = df['cidade'].map(city_map).fillna(df['cidade'])

# ----------------------------------------------------------------------
# 8. Limpar 'doenca_respiratoria'
# ----------------------------------------------------------------------
disease_map = {
    'Asma': 'Asma',
    'Bronquite': 'Bronquite',
    'Pnumonia': 'Pneumonia',
    'Cardíaco': 'Cardíaco',
    'Hipertenso': 'Hipertenso',
    'Diabético': 'Diabético'
}
df['doenca_respiratoria'] = df['doenca_respiratoria'].map(disease_map).fillna(df['doenca_respiratoria'])

# Filtra apenas doenças respiratórias
respiratory_diseases = ['Asma', 'Bronquite', 'Pneumonia']
df = df[df['doenca_respiratoria'].isin(respiratory_diseases)]

# ----------------------------------------------------------------------
# 9. Limpar 'data_do_diagnostico' (nome corrigido)
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

# Agora a coluna se chama 'data_do_diagnostico' (sem acento)
df['data_do_diagnostico'] = df['data_do_diagnostico'].apply(parse_date)
df['data_do_diagnostico'] = pd.to_datetime(df['data_do_diagnostico'], errors='coerce')
df = df.dropna(subset=['data_do_diagnostico'])
df['data_do_diagnostico'] = df['data_do_diagnostico'].dt.strftime('%d/%m/%Y')

# ----------------------------------------------------------------------
# 10. Remover duplicatas
# ----------------------------------------------------------------------
df = df.drop_duplicates(subset=['id'], keep='first')

# ----------------------------------------------------------------------
# 11. Verificação final e salvamento
# ----------------------------------------------------------------------
df.reset_index(drop=True, inplace=True)

print("\nFormato após limpeza:", df.shape)
print("\nValores ausentes por coluna:\n", df.isnull().sum())

# Salva com os nomes originais? Vamos manter os nomes padronizados para consistência
df.to_csv('dados_covid_limpo.csv', index=False, encoding='utf-8-sig')
print("\nDados limpos salvos em 'dados_covid_limpo.csv'")