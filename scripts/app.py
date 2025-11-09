#TODO: limitar inputs de gradio según el tipo de propiedad elegido

import gradio as gr
import json
import pandas as pd
import pickle
from huggingface_hub import hf_hub_download

PARAM_NAMES = {
    "rooms",
    "bedrooms",
    "bathrooms",
    "surface_total",
    "surface_covered",
    "l2",
    "property_type",
    # "surface_uncovered",
    # "available_type" #TODO: definir si la incluyo o no
}

# model_path = hf_hub_download(repo_id="alejo-cuello/bootcamp-ds-mlops-primer-ejercicio", filename="rf.pkl")
# with open(model_path, "rb") as handle:
#     model = pickle.load(handle) 

# with open("./model/categories_ohe.pkl", "rb") as handle:
#     columns_ohe = pickle.load(handle)

# with open("./model/min_max_input_values.json", "r") as handle:
#     min_max_input_values = json.load(handle)
    
def predict(*args):
    # keys = ["Age", "Class", "Wifi", "Booking", "Seat", "Checkin"]
    keys = [
        "rooms",
        "bedrooms",
        "bathrooms",
        "surface_total",
        "surface_covered",
        "l2",
        "property_type"
    ]
    # data_dict = dict(zip(keys, args))

    # single_instance = pd.DataFrame([data_dict])
    # single_instance_ohe = pd.get_dummies(single_instance,dtype="int64").reindex(columns=columns_ohe,fill_value=0)

    prediction = 100 #TODO: Borrar
    # prediction = model.predict(single_instance_ohe)

    # return ("Satisfecho" if prediction == 1 else "No Satisfecho")
    return f"U$D {prediction}"

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🏡 Estimador de precio de propiedades en venta
        """
    )
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                ---
                ## Ingrese las características que busca
                """
            )
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                ### Ambientes
                """
            )
            rooms = gr.Slider(
                label="Cantidad de ambientes",
                minimum=1,
                maximum=2,
                value=1,
                step=1
            )
            bedrooms = gr.Slider(
                label="Cantidad de dormitorios",
                minimum=1,
                maximum=2,
                value=1,
                step=1
            )
            bathrooms = gr.Slider(
                label="Cantidad de baños",
                minimum=1,
                maximum=2,
                value=1,
                step=1
            )
        with gr.Column():
            gr.Markdown(
                """
                ### Tipo y zona
                """
            )
            property_type = gr.Dropdown(
                label="Tipo de propiedad",
                choices=[
                    "Departamento",
                    "Casa",
                    "PH",
                    "Oficina",
                    "Local comercial"
                ],
                value="Departamento",
                multiselect=False
            )    
            l2 = gr.Dropdown(
                label="Zona",
                choices=[
                    "Capital Federal",
                    "Bs.As. G.B.A. Zona Norte",
                    "Bs.As. G.B.A. Zona Sur",
                    "Bs.As. G.B.A. Zona Oeste"
                ],
                value="Capital Federal",
                multiselect=False
            )
        with gr.Column():
            gr.Markdown(
                """
                ### Superficie
                """
            )
            surface_total = gr.Slider(
                label="Superficie total",
                minimum=1,
                maximum=2,
                value=1,
                step=1
            )
            surface_covered = gr.Slider(
                label="Superficie cubierta",
                minimum=1,
                maximum=2,
                value=1,
                step=1
            )
            
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                ---
                ## 💲 Precio estimado
                """
            )
            
            prediction_btn = gr.Button(value="Calcular")
            label = gr.Label(label="Score")
            prediction_btn.click(
                predict,
                inputs=[
                    l2,
                    property_type,
                    rooms,
                    bedrooms,
                    bathrooms,
                    surface_total,
                    surface_covered
                ],
                outputs=label,
                api_name="predict"
            )
    
    gr.Markdown(
        """
        <p style='text-align: center'>
            <a href='https://www.escueladedatosvivos.ai/cursos/bootcamp-de-data-science' 
                target='_blank'>Proyecto final del Bootcamp de DS y MLOps
            </a> 🤗
        </p>
        """
    )
            
demo.launch()