from gradio_client import Client

client = Client("alejo-cuello/proyecto-final-bootcamp-ds-mlops")
result = client.predict(
	param_0=3, # "Cantidad de ambientes"
 	param_1=1, # "Cantidad de dormitorios"
	param_2=1, # "Cantidad de baños"
	param_3=63, # "Superficie total (m2)"
	param_4=63, # "Superficie cubierta (m2)"
	param_5="Capital Federal", # "Zona"
	param_6="Retiro", # "Barrio"
	param_7="Departamento", # "Tipo de propiedad"
	api_name="/predict"
)
print(result)