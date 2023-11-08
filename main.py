# Librerias
from fastapi import FastAPI
import uvicorn
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections

# Import all the datasets we need for the functions performance
# Importamos todos los datasets que necesitamos para un buen desempeño de la función
dfrsSA = pd.read_csv('Dataset/mejores_business.csv')
dfcrit = pd.read_csv('Dataset/merged_stat_df.csv')

# Paso adicional para que la función corra bien.
unique_categories = dfcrit['categories'].unique()

# Create an instance o the class FastAPI
app = FastAPI()

#This is our port
#http://127.0.0.1:8000

#Presentación
@app.get("/")
def Presentacion():
    return {"Data Genius": "Bienvenido a nuestro sistema de recomendación de servicios TOP y nuevas ciudades para invertir"}

# Endpoint 1
@app.get("/Sistema_Recomendacion/{num_categoria}")
def Sistema_Recomendacion(num_categoria: int):

    """
    Basado en un área del negocio de contratistas en estados unidos ingresada por el usuario en forma de número entero,
    esta función recomienda diferentes caracteristicas que tienen las mejores empresas en esta área y las mejores ciudades
    para futura inversión.

    Las categorías son las siguientes:
    1. Appliance Repair & Installation.
    2. Cleaning Services.
    3. Interior Remodeling.
    4. HVAC & Air Quality. 
    5. Handyman or Other.
    6. Landscape & Garden Services. 
    7. Exterior Remodeling or Construction.
    8. Roofing & Insulation.
    9. Plumbing Services.
    10. Masonry & Concrete Services.
    11. Electrical Services.
    12. Home Inspection & Real Estate
    13. Art & Enterteinment

    Parámetros:
        num_categoria (int): Número de categoría que representa el área del negocio de contratistas.

    Retorna:
        dict: Un diccionario con las carácterísticas y ciudades enumeradas.
              Ejemplo: [{'Característica 1': 'Air conditioning contractor'},
                        {'Característica 2': 'Air conditioning repair service'},
                        {'Característica 3': 'Air duct cleaning service'},
                        {'Característica 4': 'Appointment required'},
                        {'Característica 5': 'HVAC contractor'},
                        {'Característica 6': 'Furnace repair service'},
                        {'Característica 7': 'Heating contractor'},
                        {'Característica 8': 'Mask required'},
                        {'Característica 9': 'Online estimates'},
                        {'Característica 10': 'Onsite services'},
                        {'Característica 11': 'Repair services'},
                        {'Característica 12': 'Staff get temperature checks'},
                        {'Característica 13': 'Staff required to disinfect surfaces between visits'},
                        {'Característica 14': 'Staff wear masks'},
                        {'Característica 15': 'Temperature check required'},
                        {'Ciudad 1': 'Brentwood, California'},
                        {'Ciudad 2': 'Gilbert, Arizona'},
                        {'Ciudad 3': 'Lisle, Illinois'},
                        {'Ciudad 4': 'Sarasota, Florida'}]
    """

    #Verificamos que el número de categoría ingresado es un número entero entre 1 y 11. No hay mejores negocios para las categorias 12
    # y 13, por lo que no se contemplan en esta función.
    if ((isinstance(num_categoria, int)) & ((num_categoria > 0) & (num_categoria < 14))):
        if ((num_categoria > 11) & (num_categoria < 14)):
            return 'Para esta categoría no hay buenos negocios que permitan ofrecer una recomendación'
        else:
            #Definimos el nombre de la categoría para filtrar en el dataframe llamado 'dfcrit' y 
            categoria = unique_categories[num_categoria-1]

            # 1. Vamos a recomendar los servicios y atributos que más se sugieren tener basados en la selección de la categoria de constratistas ingresada por el usuario.
            mask1 = (dfcrit['high_star_business'] == True) & (dfcrit['high_review_count'] == True) & (dfcrit['categories'] == categoria)
            bid_rsSA = dfcrit[mask1]
            
            # Ahora aplicamos este filtro a los mejores negocios.
            dfrsSA_filtered = dfrsSA[dfrsSA['business_id'].isin(bid_rsSA['business_id'])].copy()

            # Obtenemos la fila que contiene mayor número de palabras en la columna combSA, lo cual indica que es el business_id 
            # que ofrece más variedad de servicios y atributos. Esto permite que el sistema de recomendación sugiera negocios 
            # con mayor cantidad de servicios y atributos para que Construction Valdez los implemente. 
            # Dividir cada valor de 'combSA' en palabras y contar la cantidad de palabras
            dfrsSA_filtered['word_count'] = dfrsSA_filtered['combSA'].str.split().apply(len)

            # Encontrar la fila con el mayor número de palabras
            fila_con_mas_palabras = dfrsSA_filtered[dfrsSA_filtered['word_count'] == dfrsSA_filtered['word_count'].max()]

            # Obtenemos el business_id con más servicios y atributos para esa categoría.
            business_id = fila_con_mas_palabras.iloc[0,0]
            
            # Ahora vectorizamos las palabras de la columna dfrsSA_filtered['combSA']
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(dfrsSA_filtered['combSA'].fillna(''))

            # Establecemos el número de dimensiones en la matriz TF-IDF
            num_dimensions = tfidf_matrix.shape[1]

            # También, establecemos el número de funciones de hash (proyecciones binarias aleatorias)
            num_hash_functions = 4  # Podemos ajustar este valor según nuestras necesidades

            # Creamos el motor LSH (Localización Sensible al Hashing)
            engine = Engine(num_dimensions, lshashes=[RandomBinaryProjections('rbp', num_hash_functions)])
            
            # Recorrer cada fila y su índice en la matriz TF-IDF
            for i, row in enumerate(tfidf_matrix):

                # Extraer el 'business_id' de la fila correspondiente en el DataFrame 'dfrsSA_filtered'
                business_id1 = dfrsSA_filtered.iloc[i]['business_id']

                # Almacenar el vector TF-IDF como un arreglo aplanado en el motor LSH, asociado con el ID del negocio
                engine.store_vector(row.toarray().flatten(), data=business_id1)

            # Obtenemos el índice LSH del juego de entrada

            # Consultamos la matriz TF-IDF para obtener el vector TF-IDF del business_id de entrada
            query = tfidf_matrix[dfrsSA_filtered['business_id'] == business_id].toarray().flatten()

            # Usamos LSH para encontrar business_id similares (vecinos) al business_id de entrada
            neighbors = engine.neighbours(query)

            # Recomendaciones basadas en LSH

            # Extraemos los business_id de los business_id recomendados, excluyendo el business_id de entrada, y limitamos a los primeros 5
            recommended_business_ids = [neighbor[1] for neighbor in neighbors if neighbor[1] != business_id][:5]

            # Filtramos el DataFrame para obtener detalles de los business_id recomendados (combSA)
            recommended_business = dfrsSA_filtered[dfrsSA_filtered['business_id'].isin(recommended_business_ids)][['combSA']]

            # Construimos la lista de recomendaciones en formato JSON
            data = [{'Rec {}'.format(i + 1): business} for i, business in enumerate(recommended_business['combSA'])]

            # Inicializa una lista para almacenar las características únicas
            caracteristicas = []

            # Itera a través de los diccionarios y extrae las palabras
            for item in data:
                for value in item.values():
                    palabras = value.split("'")
                    for palabra in palabras:
                        palabra = palabra.strip()
                        if palabra and palabra != ' ':
                            caracteristicas.append(palabra)

            # Convierte la lista de características en un conjunto para eliminar duplicados
            caracteristicas_set = set(caracteristicas)

            # Convierte el conjunto nuevamente en una lista
            caracteristicas = list(caracteristicas_set)

            # Ordena la lista alfabéticamente si es necesario
            caracteristicas.sort()

            # Definir la cadena a buscar y la cadena de reemplazo
            cadena_buscar = 'C contractor'
            cadena_reemplazo = 'HVAC contractor'

            # Recorrer la lista y reemplazar 'C contractor' por 'HVAC contractor'
            for i in range(len(caracteristicas)):
                if caracteristicas[i] == cadena_buscar:
                    caracteristicas[i] = cadena_reemplazo

            # 2. Vamos a recomendar los servicios y atributos que más se sugieren tener basados en la selección de la categoria de constratistas ingresada por el usuario.
            mask1 = (dfcrit['high_category_consumption'] == True) & (dfcrit['low_category_volume'] == True) & (dfcrit['high_city_state_income'] == True) & (dfcrit['categories'] == categoria)
            bid_rsSA = dfcrit[mask1]

            # Ahora aplicamos este filtro a los mejores negocios.
            dfrsSA_filtered = dfrsSA[dfrsSA['business_id'].isin(bid_rsSA['business_id'])].copy()

            # Obtenemos la fila que contiene mayor número de palabras en la columna combSA, lo cual indica que es el business_id 
            # que ofrece más variedad de servicios y atributos. Esto permite que el sistema de recomendación sugiera negocios 
            # con mayor cantidad de servicios y atributos para que Construction Valdez los implemente. 
            # Dividir cada valor de 'combSA' en palabras y contar la cantidad de palabras
            dfrsSA_filtered['word_count'] = dfrsSA_filtered['combSA'].str.split().apply(len)

            # Encontrar la fila con el mayor número de palabras
            fila_con_mas_palabras = dfrsSA_filtered[dfrsSA_filtered['word_count'] == dfrsSA_filtered['word_count'].max()]

            # Obtenemos el business_id con más servicios y atributos para esa categoría.
            business_id = fila_con_mas_palabras.iloc[0,0]

            # Ahora vectorizamos las palabras de la columna dfrsSA_filtered['combSA']
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(dfrsSA_filtered['combSA'].fillna(''))

            # Establecemos el número de dimensiones en la matriz TF-IDF
            num_dimensions = tfidf_matrix.shape[1]

            # También, establecemos el número de funciones de hash (proyecciones binarias aleatorias)
            num_hash_functions = 4  # Podemos ajustar este valor según nuestras necesidades

            # Creamos el motor LSH (Localización Sensible al Hashing)
            engine = Engine(num_dimensions, lshashes=[RandomBinaryProjections('rbp', num_hash_functions)])

            # Recorrer cada fila y su índice en la matriz TF-IDF
            for i, row in enumerate(tfidf_matrix):

                # Extraer el 'business_id' de la fila correspondiente en el DataFrame 'dfrsSA_filtered'
                business_id1 = dfrsSA_filtered.iloc[i]['business_id']

                # Almacenar el vector TF-IDF como un arreglo aplanado en el motor LSH, asociado con el ID del negocio
                engine.store_vector(row.toarray().flatten(), data=business_id1)

            # Obtenemos el índice LSH del juego de entrada

            # Consultamos la matriz TF-IDF para obtener el vector TF-IDF del business_id de entrada
            query = tfidf_matrix[dfrsSA_filtered['business_id'] == business_id].toarray().flatten()

            # Usamos LSH para encontrar business_id similares (vecinos) al business_id de entrada
            neighbors = engine.neighbours(query)

            # Recomendaciones basadas en LSH

            # Extraemos los business_id de los business_id recomendados, excluyendo el business_id de entrada, y limitamos a los primeros 5
            recommended_business_ids = [neighbor[1] for neighbor in neighbors if neighbor[1] != business_id][:5]

            # Filtramos el DataFrame para obtener detalles de los business_id recomendados (city_state)
            recommended_business = dfrsSA_filtered[dfrsSA_filtered['business_id'].isin(recommended_business_ids)][['city_state']]

            # Construimos la lista de recomendaciones en formato JSON
            data = [{'Rec {}'.format(i + 1): business} for i, business in enumerate(recommended_business['city_state'])]

            # Inicializa una lista para almacenar las características únicas
            ciudades = []

            # Itera a través de los diccionarios y extrae las palabras
            for item in data:
                for value in item.values():
                    palabras = value.split("'")
                    for palabra in palabras:
                        palabra = palabra.strip()
                        if palabra and palabra != ' ':
                            ciudades.append(palabra)

            # Convierte la lista de características en un conjunto para eliminar duplicados
            ciudades_set = set(ciudades)

            # Convierte el conjunto nuevamente en una lista
            ciudades = list(ciudades_set)

            # Ordena la lista alfabéticamente si es necesario
            ciudades.sort()

            # Crea un diccionario con la clave 'Características Recomendadas'
            #dic_resultado = {'Características Recomendadas': caracteristicas, 'Ciudades Recomendadas': ciudades}

            #return print(dic_resultado)

            # Crear una lista de diccionarios para las características
            caracteristicas_formateadas = [{'Característica {}'.format(i + 1): caracteristica}
                                        for i, caracteristica in enumerate(caracteristicas)]

            # Crear una lista de diccionarios para las ciudades
            ciudades_formateadas = [{'Ciudad {}'.format(i + 1): ciudad}
                                    for i, ciudad in enumerate(ciudades)]

            # Combinar las listas de características y ciudades formateadas en una sola lista
            resultado_formateado = caracteristicas_formateadas + ciudades_formateadas

            return resultado_formateado
    else:
        #Mandamos un mensaje de error.
        return 'Por favor ingrese un número entero entre el 1 y el 13'