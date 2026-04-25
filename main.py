from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load('model/modelo_prestamos.joblib')
model_v2 = joblib.load('model/modelo_prestamos_v2.joblib')


class PrestamoInput(BaseModel):
    edad: int
    ingresos_anuales: float
    score_crediticio: int
    deuda_total: float


@app.post("/evaluar-prestamo")
def predict(data: PrestamoInput):
    # Convertimos el input a la lista que espera el modelo
    input_vars = [[data.edad, data.ingresos_anuales, data.score_crediticio, data.deuda_total]]

    prediction = model.predict(input_vars)
    probabilidad = model.predict_proba(input_vars)[0][1]  # Probabilidad de ser "1"

    resultado = "Aprobado" if prediction[0] == 1 else "Rechazado"
    
    result_v1 = {
        "status": resultado,
        "score_aprobacion": round(probabilidad * 100, 2),
        "mensaje": f"El crédito fue {resultado.lower()} con un {round(probabilidad * 100)}% de confianza."
    }
    #v2
    dti                    = data.deuda_total / data.ingresos_anuales
    capacidad_ahorro       = data.ingresos_anuales - data.deuda_total
    score_crediticio_edad  = data.score_crediticio / data.edad
    apalancamiento_critico = int(data.deuda_total > (data.ingresos_anuales * 0.05))
    multiplicador_estabilidad = (data.score_crediticio * data.edad) / 100
    input_vars_v2 = [[
        data.edad,
        data.ingresos_anuales,
        data.score_crediticio,
        data.deuda_total,
        dti,
        capacidad_ahorro,
        score_crediticio_edad,
        apalancamiento_critico,
        multiplicador_estabilidad
    ]]
    prediction_v2 = model_v2.predict(input_vars_v2)
    probabilidad_v2 = model_v2.predict_proba(input_vars_v2)[0][1]  # Probabilidad de ser "1"

    resultado_v2 = "Aprobado" if prediction_v2[0] == 1 else "Rechazado"
    
    result_v2 = {
        "status": resultado_v2,
        "score_aprobacion": round(probabilidad_v2 * 100, 2),
        "mensaje": f"El crédito fue {resultado_v2.lower()} con un {round(probabilidad_v2 * 100)}% de confianza."
    }
    return {
        "modelo_v1": result_v1,
        "modelo_v2": result_v2
    }