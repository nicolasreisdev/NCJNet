from fastapi import FastAPI
import uvicorn 
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder


# Criar um instância no FasAPI
app = FastAPI()

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
    'Historia': 'Humanas',
    'Engenharia Eletrica': 'Exatas', 
    'Engenharia da Computação': 'Tecnologia',
    'Engenharia Civil': 'Exatas',
    'Enfermagem': 'Saúde',
    'Biologia': 'Biológicas',
    'Ciencia da Computação': 'Tecnologia',
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
    'Quimica': 'Exatas',
    'Fisica': 'Exatas',
    'Matematica': 'Exatas',
    'Letras': 'Linguagens',
    'Danca': 'Artes',
    'Musica': 'Artes',
    'Teatro': 'Artes',
    'Artes Cênicas': 'Artes',
    'Artes Plásticas': 'Artes',
    'Cinema': 'Artes',
    'Sistemas da Informação': 'Tecnologia',
    'Agronomia': 'Biológicas',
    'Historia': 'Humanas',
    'Ciencias Sociais': 'Humanas',
    'Servico Social': 'Humanas',
    'Educacao Fisica': 'Saúde',
    'Relacoes Internacionais': 'Humanas',
    'Jornalismo': 'Comunicacao',
    'Publicidade e Propaganda': 'Comunicacao',
    'Relações Públicas': 'Comunicacao',
    'Marketing': 'Comunicacao',
    'Administração': 'Humanas',
    'Farmacia': 'Biológicas',
    'Biomedicina': 'Biológicas',
    'Nutricao': 'Biológicas',
    'Fisioterapia': 'Saúde',
    'Fonoaudiologia': 'Saúde',
    'Analise e Desenvolvimento de Sistemas': 'Tecnologia',
    'Ciencia de Dados': 'Tecnologia',
    'Linguistica': 'Linguagens',
    'Pedagogia': 'Humanas',
    'Engenharia Florestal': 'Biologicas',
    'Moda': 'Artes',
    'Design Grafico': 'Artes',
    'Design de Interiores': 'Artes',
    'Museologia': 'Humanas',
    'Gestao Ambiental': 'Exatas',
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
        'Artes': data.notaArtes
    }
    feature_cols = ['Matematica', 'Portugues', 'Literatura', 'Redacao', 'Quimica', 'Fisica', 'Biologia', 'Geografia', 'Historia', 'Filosofia', 'Sociologia', 'Artes']
    pred_df = pd.DataFrame([input_features], columns=feature_cols)
    
    y = sugestorModel.predict_proba(pred_df)[0]
    
    recomendacoes_area = []
    for i, nome_curso in enumerate(label_curso.classes_):
        if areas_cursos.get(nome_curso) == data.areaPreferencia:
            recomendacoes_area.append({
                'curso': nome_curso,
                'probabilidade_aptidao': y[i],
                'area': data.areaPreferencia
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
    