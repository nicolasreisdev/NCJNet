from fastapi import FastAPI
import unicodedata

import uvicorn 
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os



# Criar um instância no FasAPI
app = FastAPI()

def normalize(s):
    return unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('ASCII').strip()

# Criar uma classe com os dados de entrada que virão no body da requisição com os tipos esperados
class request_body(BaseModel):
    notaMatematica: int
    notaPortugues: int
    notaLiteratura: int
    notaRedacao: int
    notaQuimica: int
    notaFisica: int
    notaBiologia: int
    notaGeografia: int
    notaHistoria: int
    notaFilosofia: int
    notaSociologia: int
    notaArtes: int
    areaPreferencia: str
    
# Carregar modelo para realizar a predição
# sugestorModel = joblib.load('./model/sugestor.pkl')
# sugestorModel = joblib.load('./model/sugestorXGB.pkl')
areas_cursos = joblib.load('./model/areaCursos.pkl')
scaler = joblib.load('./model/scaler.pkl')
""" areas_cursos = {
    'Sociologia': 'Humanas',
    'Psicologia': 'Humanas',
    'Odontologia': 'Saúde',
    'Medicina Veterinária': 'Biológicas',
    'Medicina': 'Saúde',
    'Engenharia Elétrica': 'Exatas', 
    'Engenharia da Computação': 'Tecnologia',
    'Engenharia Civil': 'Exatas',
    'Enfermagem': 'Saúde',
    'Biologia': 'Biológicas',
    'Ciência da Computação': 'Tecnologia',
    'Arquitetura e Urbanismo': 'Exatas',
    'Direito': 'Humanas',
    'Design': 'Exatas',
    'Artes Visuais': 'Artes',
    'Engenharia de Produção': 'Exatas',
    'Engenharia Mecânica': 'Exatas',
    'Engenharia Química': 'Exatas',
    'Engenharia Ambiental': 'Exatas',
    'Engenharia de Alimentos': 'Exatas',
    'Engenharia de Software': 'Tecnologia',
    'Geografia': 'Humanas',
    'Filosofia': 'Humanas',
    'Química': 'Exatas',
    'Física': 'Exatas',
    'Matemática': 'Exatas',
    'Letras': 'Linguagens',
    'Dança': 'Artes',
    'Música': 'Artes',
    'Teatro': 'Artes',
    'Artes Cênicas': 'Artes',
    'Artes Plásticas': 'Artes',
    'Cinema': 'Artes',
    'Sistemas da Informação': 'Tecnologia',
    'Agronomia': 'Biológicas',
    'História': 'Humanas',
    'Ciências Sociais': 'Humanas',
    'Serviço Social': 'Humanas',
    'Educação Física': 'Saúde',
    'Relações Internacionais': 'Humanas',
    'Jornalismo': 'Comunicação',
    'Publicidade e Propaganda': 'Comunicação',
    'Relações Públicas': 'Comunicação',
    'Marketing': 'Comunicação',
    'Administração': 'Humanas',
    'Farmácia': 'Biológicas',
    'Biomedicina': 'Biológicas',
    'Nutrição': 'Biológicas',
    'Fisioterapia': 'Saúde',
    'Fonoaudiologia': 'Saúde',
    'ADS': 'Tecnologia',
    'Ciência de Dados': 'Tecnologia',
    'Linguística': 'Linguagens',
    'Pedagogia': 'Humanas',
    'Engenharia Florestal': 'Biológicas',
    'Moda': 'Artes',
    'Design Grafico': 'Artes',
    'Design de Interiores': 'Artes',
    'Museologia': 'Humanas',
    'Gestão Ambiental': 'Exatas',
    'Biblioteconomia': 'Humanas'
} """

MODEL_DIR = './model/'
modelos_por_area = {}
labels_por_area = {}

for filename in os.listdir(MODEL_DIR):
    if filename.endswith('_sugestor.pkl'):
        area_name = filename.replace('_sugestor.pkl', '')
        model_path = os.path.join(MODEL_DIR, filename)
        modelos_por_area[area_name] = joblib.load(model_path)
        print(f"Modelo para a área '{area_name}' carregado com sucesso.")
        
for filename in os.listdir(MODEL_DIR):
    if filename.endswith('_labelCurso.pkl'):
        area_name = filename.replace('_labelCurso.pkl', '')
        model_path = os.path.join(MODEL_DIR, filename)
        labels_por_area[area_name] = joblib.load(model_path)
        print(f"Label para a área '{area_name}' carregado com sucesso.")
    



@app.post('/predict')
def predict(data: request_body):
    input_features = {
        'Matematica': data.notaMatematica,
        'Portugues': data.notaPortugues,
        'Literatura': data.notaLiteratura,
        'Redacao': data.notaRedacao,
        'Quimica': data.notaQuimica,
        'Fisica': data.notaFisica,
        'Biologia': data.notaBiologia,
        'Geografia': data.notaGeografia,
        'Historia': data.notaHistoria,
        'Filosofia': data.notaFilosofia,
        'Sociologia': data.notaSociologia,
        'Artes': data.notaArtes,
        'Area de Preferencia': data.areaPreferencia
    }
    
    feature_cols = ['Matematica', 'Portugues', 'Literatura', 'Redacao', 'Quimica', 'Fisica', 'Biologia', 'Geografia', 'Historia', 'Filosofia', 'Sociologia', 'Artes']

    pred_df_numeric = pd.DataFrame([
        [
            data.notaMatematica, data.notaPortugues, data.notaLiteratura,
            data.notaRedacao, data.notaQuimica, data.notaFisica,
            data.notaBiologia, data.notaGeografia, data.notaHistoria,
            data.notaFilosofia, data.notaSociologia, data.notaArtes
        ]
    ], columns=feature_cols)
    
    
    X_numeric_scaled = scaler.transform(pred_df_numeric.values)
    preferencia_normalizada = normalize(data.areaPreferencia)

    print(f"Área de Preferência recebida (original): {data.areaPreferencia}")
    print(f"Área de Preferência normalizada: {preferencia_normalizada}")

    selected_model = modelos_por_area.get(normalize(data.areaPreferencia))
    
    y = selected_model.predict_proba(X_numeric_scaled)[0]
    print(y)
    
    recomendacoes_area = []
    
    # Normalizar áreas do dicionário para evitar erros de comparação
    areas_cursos_normalizado = {normalize(k): v for k, v in areas_cursos.items()}

    labels = labels_por_area.get(normalize(data.areaPreferencia))
    
    
    if len(y) != len(labels.classes_):
        print(f"Inconsistência detectada para a área '{preferencia_normalizada}':")
        print(f"Probabilidades retornadas: {len(y)}, Cursos esperados: {len(labels.classes_)}")
        # Ajustar para usar apenas os cursos correspondentes às probabilidades
        labels.classes_ = labels.classes_[::-1]
        labels.classes_ = labels.classes_[:y.shape[0]]
    
    recomendacoes_area = []
    for i, nome_curso in enumerate(labels.classes_):
        nome_curso_limpo = nome_curso.strip()
        area_curso = areas_cursos_normalizado.get(normalize(nome_curso_limpo))
        print(f"Curso: {nome_curso_limpo}, Área do curso: {area_curso}, Área preferida: {preferencia_normalizada}")
        if area_curso and normalize(area_curso) == preferencia_normalizada:
            recomendacoes_area.append({
                'curso': nome_curso_limpo,
                'probabilidade_aptidao': float(y[i]),
                'area': area_curso
            })
    if preferencia_normalizada == "Linguagens":
        recomendacoes_area[0]['probabilidade_aptidao'] = recomendacoes_area[0]['probabilidade_aptidao']-0.05
        recomendacoes_area[1]['probabilidade_aptidao'] = recomendacoes_area[1]['probabilidade_aptidao']-0.05
        recomendacoes_area.append(
            {
                'curso': 'Línguas Estrangeiras',
                'probabilidade_aptidao': 0.10,
                'area': 'Linguagens'
            }
        )
    # Ordenar os cursos da área pela probabilidade de aptidão (do maior para o menor)
    recomendacoes_ordenadas = sorted(recomendacoes_area, key=lambda x: x['probabilidade_aptidao'], reverse=True)

    return recomendacoes_ordenadas[:3]
    