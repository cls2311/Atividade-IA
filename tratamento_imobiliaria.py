import pandas as pd
import numpy as np
import re
from datetime import datetime

# ----------------------------------------------------------------------
# 1. Carregar o conjunto de dados
# ----------------------------------------------------------------------
df = pd.read_csv('Imobiliaria_prova.csv', encoding='utf-8')
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

# Renomear colunas para garantir que os nomes fiquem conforme esperado
df.rename(columns={
    'tamanho_m': 'tamanho_m2',           # o '²' foi removido, renomeamos para m2
    'data_da_ultima_locacao': 'data_locacao'   # simplificar para evitar confusão
}, inplace=True)

print("\nColunas padronizadas:", df.columns.tolist())

# ----------------------------------------------------------------------
# 3. Padronizar 'localizacao' (cidades)
# ----------------------------------------------------------------------
city_map = {
    'João Pessoa': 'João Pessoa',
    'JP': 'João Pessoa',
    'Campina Grande': 'Campina Grande',
    'CG': 'Campina Grande',
    'Patos': 'Patos',
    'PT': 'Patos',
    'Cajazeiras': 'Cajazeiras',
    'Cabedelo': 'Cabedelo',
    'Sousa': 'Sousa',
    'Pombal': 'Pombal',
    'Riacho dos cavalos': 'Riacho dos Cavalos'  # corrige capitalização
}
df['localizacao'] = df['localizacao'].map(city_map).fillna(df['localizacao'])

# Imputar cidades ausentes pela moda
moda_cidade = df['localizacao'].mode()[0] if not df['localizacao'].mode().empty else 'João Pessoa'
df['localizacao'] = df['localizacao'].fillna(moda_cidade)

# ----------------------------------------------------------------------
# 4. Padronizar 'tipo_de_imovel' e remover tipos que não são imóveis
# ----------------------------------------------------------------------
tipo_map = {
    'Casa': 'Casa',
    'Apartamento': 'Apartamento',
    'Kitnet': 'Kitnet',
    'Duplex': 'Duplex',
    'Galpão': 'Galpão',
    'Quarto': 'Quarto',               # embora seja cômodo, pode ser considerado imóvel
    'Veículo usado': np.nan,          # não é imóvel → remover
    'Carro': np.nan                   # não é imóvel → remover
}
df['tipo_de_imovel'] = df['tipo_de_imovel'].map(tipo_map).fillna(df['tipo_de_imovel'])

# Remove linhas onde o tipo é inválido (NaN após mapeamento)
df = df.dropna(subset=['tipo_de_imovel'])

# ----------------------------------------------------------------------
# 5. Limpar 'data_locacao'
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

df['data_locacao'] = df['data_locacao'].apply(parse_date)
df['data_locacao'] = pd.to_datetime(df['data_locacao'], errors='coerce')
df = df.dropna(subset=['data_locacao'])   # remove registros sem data
df['data_locacao'] = df['data_locacao'].dt.strftime('%d/%m/%Y')

# ----------------------------------------------------------------------
# 6. Limpar 'comodos' (máximo 7)
# ----------------------------------------------------------------------
df['comodos'] = pd.to_numeric(df['comodos'], errors='coerce')
df.loc[df['comodos'] > 7, 'comodos'] = np.nan   # inválido se > 7
mediana_comodos = df['comodos'].median()
df['comodos'] = df['comodos'].fillna(mediana_comodos)

# ----------------------------------------------------------------------
# 7. Limpar 'tamanho_m2'
# ----------------------------------------------------------------------
df['tamanho_m2'] = pd.to_numeric(df['tamanho_m2'], errors='coerce')
df.loc[df['tamanho_m2'] <= 0, 'tamanho_m2'] = np.nan   # zero ou negativo é inválido
mediana_m2 = df['tamanho_m2'].median()
df['tamanho_m2'] = df['tamanho_m2'].fillna(mediana_m2)

# ----------------------------------------------------------------------
# 8. Limpar 'valor_da_locacao' (de R$ 800 a R$ 1500)
# ----------------------------------------------------------------------
# Remove "R$", vírgulas e converte para float
def parse_valor(val):
    if pd.isna(val):
        return np.nan
    val = str(val).replace('R$', '').replace('.', '').replace(',', '.').strip()
    try:
        return float(val)
    except:
        return np.nan

df['valor_da_locacao'] = df['valor_da_locacao'].apply(parse_valor)
df.loc[(df['valor_da_locacao'] < 800) | (df['valor_da_locacao'] > 1500), 'valor_da_locacao'] = np.nan
mediana_valor = df['valor_da_locacao'].median()
df['valor_da_locacao'] = df['valor_da_locacao'].fillna(mediana_valor)

# ----------------------------------------------------------------------
# 9. Padronizar 'vaga_na_garagem' (Sim/Não)
# ----------------------------------------------------------------------
vaga_map = {
    'Sim': 'Sim',
    'S': 'Sim',
    'Não': 'Não',
    'N': 'Não',
    np.nan: np.nan
}
df['vaga_na_garagem'] = df['vaga_na_garagem'].map(vaga_map).fillna(df['vaga_na_garagem'])
moda_vaga = df['vaga_na_garagem'].mode()[0] if not df['vaga_na_garagem'].mode().empty else 'Não'
df['vaga_na_garagem'] = df['vaga_na_garagem'].fillna(moda_vaga)

# ----------------------------------------------------------------------
# 10. Remover duplicatas baseadas no código do imóvel
# ----------------------------------------------------------------------
# A primeira coluna já está como 'codigo_imovel' (nome padronizado)
df = df.drop_duplicates(subset=['codigo_imovel'], keep='first')

# ----------------------------------------------------------------------
# 11. Verificação final e salvamento
# ----------------------------------------------------------------------
df.reset_index(drop=True, inplace=True)

print("\nFormato após limpeza:", df.shape)
print("\nValores ausentes por coluna:\n", df.isnull().sum())

df.to_csv('imobiliaria_limpo.csv', index=False, encoding='utf-8-sig')
print("\nDados limpos salvos em 'imobiliaria_limpo.csv'")