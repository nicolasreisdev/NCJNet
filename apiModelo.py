from fastapi import FastAPI
import unicodedata

import uvicorn 
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder



# Criar um instância no FasAPI
app = FastAPI()

def normalize(s):
    return unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('ASCII').lower().strip()

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
sugestorModel = joblib.load('./model/sugestorXGB.pkl')
label_curso = joblib.load('./model/labelCurso.pkl')
areas_cursos = {
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
    'Jornalismo': 'Comunicacao',
    'Publicidade e Propaganda': 'Comunicacao',
    'Relações Públicas': 'Comunicacao',
    'Marketing': 'Comunicacao',
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
    'Engenharia Florestal': 'Biologicas',
    'Moda': 'Artes',
    'Design Grafico': 'Artes',
    'Design de Interiores': 'Artes',
    'Museologia': 'Humanas',
    'Gestão Ambiental': 'Exatas',
    'Biblioteconomia': 'Humanas'
}



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
        '"Area de Preferencia"': data.areaPreferencia
    }
    
    encoder = joblib.load('./model/encoderAreaPref.pkl')
    feature_cols = ['Matematica', 'Portugues', 'Literatura', 'Redacao', 'Quimica', 'Fisica', 'Biologia', 'Geografia', 'Historia', 'Filosofia', 'Sociologia', 'Artes','"Area de Preferencia"']

    pred_df = pd.DataFrame([input_features], columns=feature_cols)


    area_pref_encoded = encoder.transform(pred_df[['"Area de Preferencia"']])
    area_pref_encoded = area_pref_encoded.astype(float) * 5
    
    X_numeric = pred_df.drop(columns=['"Area de Preferencia"'])
    X_pred = np.concatenate([X_numeric.values, area_pref_encoded], axis=1)
    
    y = sugestorModel.predict_proba(X_pred)[0]
    
    preferencia_normalizada = normalize(data.areaPreferencia)
    print(list(label_curso.classes_))
    
    print("Preferência normalizada recebida:", preferencia_normalizada)
    recomendacoes_area = []
    for i, nome_curso in enumerate(label_curso.classes_):
        area_curso = areas_cursos.get(nome_curso)
        print(f"{nome_curso}: {area_curso} -> {normalize(area_curso) if area_curso else None}")
        if area_curso and normalize(area_curso) == preferencia_normalizada:
            recomendacoes_area.append({
                'curso': nome_curso,
                'probabilidade_aptidao': y[i],
                'area': area_curso
            })

    # Ordenar os cursos da área pela probabilidade de aptidão (do maior para o menor)
    recomendacoes_ordenadas = sorted(recomendacoes_area, key=lambda x: x['probabilidade_aptidao'], reverse=True)

    return recomendacoes_ordenadas[:3]
    
# def predict(data: request_body):
#     input_features = {
#     'tempo_na_empresa': data.tempo_na_empresa,
#     'nivel_na_empresa': data.nivel_na_empresa
#     }

#     pred_df = pd.DataFrame(input_features, index=[1])
    
#     # Predição
#     y_pred = model_poly.predict(pred_df)[0].astype(float)
    
#     return {'salario_em_reais': y_pred.tolist()}
    