from statistics import stdev
import traceback
import numpy as np
import pandas as pd
import category_encoders as ce
import statistics as stats
import warnings
warnings.filterwarnings("ignore")
import sklearn
import math
import pickle
import joblib
from joblib import load
from flask import Flask, request, jsonify

app = Flask(__name__)

columnas_produccion = load('columnas_produccion.joblib')
rf = load('rf_hiper.joblib')

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    # Cargamos el json y lo convertimos en dataframe para trabajar con el
    json_ = request.get_json(force = True)
    
    df = pd.DataFrame.from_dict(json_)

    # Cargamos las columnas tras el mean encoding ya que nos surgían problemas al hacerlo aquí directamente
    
    C_CONF_encod = load('encod_C_CONF.joblib')
    C_RCFG_encod = load('encod_C_RCFG.joblib')
    C_VEHS_encod = load('encod_C_VEHS.joblib')
    P_SEX_encod = load('encod_P_SEX.joblib')
    C_WTHR_encod = load('encod_C_WTHR.joblib')
    C_RALN_encod = load('encod_C_RALN.joblib')
    C_TRAF_encod = load('encod_C_TRAF.joblib')
    V_TYPE_encod = load('encod_V_TYPE.joblib')
    P_AGE_encod = load('encod_P_AGE.joblib')
    P_PSN_encod = load('encod_P_PSN.joblib')
    P_SAFE_encod = load('encod_P_SAFE.joblib')
    C_V_YEARS_encod = load('encod_C_V_YEARS.joblib')
    
    #Una vez cargadas, las renombramos en el dataframe
    
    df['C_CONF'] = df['C_CONF'].map(C_CONF_encod)
    df['C_RCFG'] = df['C_RCFG'].map(C_RCFG_encod)
    df['C_VEHS']= df['C_VEHS'].map(C_VEHS_encod)
    df['P_SEX'] = df['P_SEX'].map(P_SEX_encod)
    df['C_WTHR'] = df['C_WTHR'].map(C_WTHR_encod)
    df['C_RALN'] = df['C_RALN'].map(C_RALN_encod)
    df['C_TRAF'] = df['C_TRAF'].map(C_TRAF_encod)
    df['V_TYPE'] = df['V_TYPE'].map(V_TYPE_encod)
    df['P_AGE'] = df['P_AGE'].map(P_AGE_encod)
    df['P_PSN'] = df['P_PSN'].map(P_PSN_encod)
    df['P_SAFE'] = df['P_SAFE'].map(P_SAFE_encod)
    
    df['C_V_YEARS']= df['C_YEAR']- df['V_YEAR']
    df['C_V_YEARS'] = df['C_V_YEARS'].astype('float')
    df = df.drop(['V_YEAR', 'C_YEAR'], axis=1)
    
    df['C_V_YEARS'] = df['C_V_YEARS'].map(C_V_YEARS_encod)
        
    
    # Ahora hacemos el ciclical encoding para las variables temporales y borramos las columnas originales
    
    time_columns = ['C_HOUR', 'C_MNTH', 'C_WDAY']
    
    def codificacion_ciclica(dataset, columns):
        for columna in time_columns:
            dataset[columna+"_norm"] = 2*math.pi*dataset[columna]/dataset[columna].max()
            dataset["cos_"+columna] = np.cos(dataset[columna+"_norm"])
            dataset["sin_"+columna] = np.sin(dataset[columna+"_norm"])
            dataset = dataset.drop([columna+"_norm"], axis=1)
        return dataset

    df['C_HOUR']= df['C_HOUR'].astype('float')
    df['C_MNTH']= df['C_MNTH'].astype('float')
    df['C_WDAY']= df['C_WDAY'].astype('float')

    df = codificacion_ciclica(df, time_columns)

    for i in time_columns:
        df = (df.drop(i, axis=1))

    #Elegimos las columnas que van a entrar al modelo y, por tanto que debemos escalar    
    columnas_modelo=['C_VEHS', 'C_CONF', 'C_RCFG',
            'C_WTHR', 'C_RALN', 'C_TRAF', 'V_TYPE',
            'P_SEX', 'P_AGE', 'P_PSN', 'P_SAFE', 'C_V_YEARS', 'cos_C_HOUR',
            'sin_C_HOUR', 'cos_C_MNTH', 'sin_C_MNTH', 'cos_C_WDAY', 'sin_C_WDAY']
    
    df = df.astype('float')

    #Cargamos el escalado y lo llevamos a cabo para las columnas que hemos elegido

    scaler = load('model_scaled.joblib')
    collision_scaled = pd.DataFrame(scaler.transform(df), columns=columnas_modelo)
    
    #Hacemos las predicciones
    prediction = rf.predict_proba(collision_scaled)
    y_pred_best = prediction[:, 1]

    return jsonify({'La_probabilidad_de_fallecimiento_es': str(y_pred_best)})

if __name__ == "__main__":
    try:
    #Elegimos el puerto
        port = int(sys.argv[1]) 
    except:
        port = 80  
    # Cargamos el modelo de hiperparámetros
    modelo = joblib.load("rf_hiper.pkl")
    print('Model loaded')
    # Cargamos las columnas que utilizamos para la prediccion
    model_columns = joblib.load("columnas_produccion.joblib")  
    print('Model columns loaded')

    app.run(port=port, debug=True)
