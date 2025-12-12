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
    "l3",
    "property_type"
}

model_path = "mlruns/512582443179615027/models/m-454cb18b3a6d4aa3b4f229257f7fb5c8/artifacts/model.pkl"

with open(model_path, "rb") as handle:
    model = pickle.load(handle)
    
with open("notebooks/categories_ohe.pkl", "rb") as handle:
    columns_ohe = pickle.load(handle)

with open("notebooks/min_max_input_values.json", "rb") as handle:
    min_max_input_values = json.load(handle)

with open("notebooks/l3_by_l2.json", "rb") as handle:
    l3_by_l2 = json.load(handle)
    
def predict(*args):
    keys = [
        "rooms",
        "bedrooms",
        "bathrooms",
        "surface_total",
        "surface_covered",
        "l2",
        "l3",
        "property_type",
        # "lat", # Voy a sacar latitud y longitud porque no son datos que el usuario pueda ingresar desde la interfaz de Gradio
        # "lon" # Voy a sacar latitud y longitud porque no son datos que el usuario pueda ingresar desde la interfaz de Gradio
    ]
    
    data_dict = dict(zip(keys, args))
    single_instance = pd.DataFrame([data_dict])
    
    columns_ohe_2 = columns_ohe.drop(["lat","lon","available_publication","days_since_start","days_since_end"], errors="ignore")
    
    single_instance_ohe = pd.get_dummies(single_instance,dtype="int64").reindex(columns=columns_ohe_2,fill_value=0)

    prediction = model.predict(single_instance_ohe)

    return f"U$D {round(prediction[0],2)}"

def update_l3_selector(l2):

    if l2 is None:
        return gr.Dropdown(
            label="Barrio",
            choices=[],
            value=None,
            multiselect=False
        )

    return gr.Dropdown(
        label="Barrio",
        choices=sorted(l3_by_l2[l2]),
        value=l3_by_l2[l2][0],
        multiselect=False
    )

def update_surface_covered(max_total):
    if max_total is None:
        return gr.Slider(
            label="Superficie cubierta (m2)",
            minimum=min_max_input_values["surface_covered"]["Min"],
            maximum=min_max_input_values["surface_covered"]["Max"],
            value=min_max_input_values["surface_covered"]["Min"],
            step=1
        )

    return gr.Slider(
        label="Superficie cubierta (m2)",
        minimum=min_max_input_values["surface_covered"]["Min"],
        maximum=max_total,
        value=max_total,
        step=1
    )

def update_bedrooms_bathrooms(max_total):
    if max_total is None:
        return (
            gr.Slider(
                label="Cantidad de dormitorios",
                minimum=min_max_input_values["bedrooms"]["Min"],
                maximum=min_max_input_values["bedrooms"]["Max"],
                value=min_max_input_values["bedrooms"]["Min"],
                step=1
            ),
            gr.Slider(
                label="Cantidad de baños",
                minimum=min_max_input_values["bathrooms"]["Min"],
                maximum=min_max_input_values["bathrooms"]["Max"],
                value=min_max_input_values["bathrooms"]["Min"],
                step=1
            )
        )

    max_bedrooms = min_max_input_values["bedrooms"]["Max"] if min_max_input_values["bedrooms"]["Max"] < max_total else max_total 
    max_bathrooms = min_max_input_values["bathrooms"]["Max"] if min_max_input_values["bathrooms"]["Max"] < max_total else max_total

    return (
        gr.Slider(
            label="Cantidad de dormitorios",
            minimum=min_max_input_values["bedrooms"]["Min"],
            maximum=max_bedrooms,
            value=min_max_input_values["bedrooms"]["Min"],
            step=1
        ),
        gr.Slider(
            label="Cantidad de baños",
            minimum=min_max_input_values["bathrooms"]["Min"],
            maximum=max_bathrooms,
            value=min_max_input_values["bathrooms"]["Min"],
            step=1
        )
    )

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
                ## Ingrese las características que busca
                """
            )
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                ### Tipo y zona
                """
            )
            property_type = gr.Dropdown(
                label="Tipo de propiedad",
                choices=sorted([
                    "Departamento",
                    "Casa",
                    "PH",
                    "Oficina",
                    "Local comercial"
                ]),
                value="Departamento",
                multiselect=False
            )    
            l2 = gr.Dropdown(
                label="Zona",
                choices=sorted(list(l3_by_l2.keys())),
                value="Capital Federal",
                multiselect=False
            )
            l3 = gr.Dropdown(
                label="Barrio",
                choices=sorted(l3_by_l2["Capital Federal"]),
                value=l3_by_l2["Capital Federal"][0],
                multiselect=False
            )
            l2.change(
                fn=update_l3_selector,
                inputs=l2,
                outputs=l3
            )
        with gr.Column():
            gr.Markdown(
                """
                ### Ambientes
                """
            )
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
            rooms.change(
                fn=update_bedrooms_bathrooms,
                inputs=rooms,
                outputs=[bedrooms, bathrooms]
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
            surface_total.change(
                fn=update_surface_covered,
                inputs=surface_total,
                outputs=surface_covered
            )
            
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
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
                    l3,
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