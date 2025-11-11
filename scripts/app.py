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
    "property_type"
}

# model_path = hf_hub_download(repo_id="alejo-cuello/bootcamp-ds-mlops-primer-ejercicio", filename="rf.pkl")
model_path = "mlruns/512582443179615027/models/m-0248ec91bbc349f393da1c30e4f3fed1/artifacts/model.pkl"

with open(model_path, "rb") as handle:
    model = pickle.load(handle)
    
with open("notebooks/categories_ohe.pkl", "rb") as handle:
    columns_ohe = pickle.load(handle)

with open("notebooks/min_max_input_values.json", "rb") as handle:
    min_max_input_values = json.load(handle)
    
def predict(*args):
    keys = [
        "rooms",
        "bedrooms",
        "bathrooms",
        "surface_total",
        "surface_covered",
        "l2",
        "property_type",
        # "lat",
        # "lon"
    ]
    
    data_dict = dict(zip(keys, args))
    single_instance = pd.DataFrame([data_dict])
    
    #TODO: Sería bueno mostrar un mapa para obtener del cliente una latitud y longitud. O sino hacer un modelo si estos inputs 
    # De momento dejo hardcodeado un valor promedio
    single_instance["lat"] =  -58.46
    single_instance["lon"] =  -34.6
    
    single_instance_ohe = pd.get_dummies(single_instance,dtype="int64").reindex(columns=columns_ohe,fill_value=0)
    print("ALE",single_instance_ohe.drop(columns=["property_type_Departamento","property_type_Local comercial","property_type_Oficina","property_type_PH"]))
    prediction = model.predict(single_instance_ohe)

    return f"U$D {round(prediction[0],2)}"

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
            #TODO: Se podrían mostrar cantidades de ambientes condicionales al tipo de propiedad seleccionado
            rooms = gr.Slider(
                label="Cantidad de ambientes",
                minimum=min_max_input_values["rooms"]["Min"],
                maximum=min_max_input_values["rooms"]["Max"],
                value=min_max_input_values["rooms"]["Min"],
                step=1
            )
            bedrooms = gr.Slider(
                label="Cantidad de dormitorios",
                minimum=min_max_input_values["bedrooms"]["Min"],
                maximum=min_max_input_values["bedrooms"]["Max"],
                value=min_max_input_values["bedrooms"]["Min"],
                step=1
            )
            bathrooms = gr.Slider(
                label="Cantidad de baños",
                minimum=min_max_input_values["bathrooms"]["Min"],
                maximum=min_max_input_values["bathrooms"]["Max"],
                value=min_max_input_values["bathrooms"]["Min"],
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
                label="Superficie total (m2)",
                minimum=min_max_input_values["surface_total"]["Min"],
                maximum=min_max_input_values["surface_total"]["Max"],
                value=min_max_input_values["surface_total"]["Min"],
                step=1
            )
            surface_covered = gr.Slider(
                label="Superficie cubierta (m2)",
                minimum=min_max_input_values["surface_covered"]["Min"],
                maximum=min_max_input_values["surface_covered"]["Max"],
                value=min_max_input_values["surface_covered"]["Min"],
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
                    rooms,
                    bedrooms,
                    bathrooms,
                    surface_total,
                    surface_covered,
                    l2,
                    property_type,
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