from genericpath import samestat
import numpy as np
import pandas as pd
from pandas.io.formats import style
import plotly.graph_objects as go
import plotly.express as px

import dash
import dash_daq as daq
from dash import html
from dash import dcc
import joblib
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

# Diego
#df_listings = pd.read_csv("/Users/diegoma/kaggle/listings.csv")
#df_neighbourhoods = pd.read_csv("/Users/diegoma/kaggle/neighbourhoods.csv")
#df_modelo = pd.read_csv("/Users/diegoma/kaggle/datos_modelo.csv")

# Jaime
#df_listings = pd.read_csv(r"C:\Users\jaime\Documents\ICAI\Quinto\PrácticaFinal_BI_EstadisticaComp\AirbnbData\CleanData\df_listings_detailed_limpio.csv")
#df_neighbourhoods = pd.read_csv(r"C:\Users\jaime\Documents\ICAI\Quinto\PrácticaFinal_BI_EstadisticaComp\AirbnbData\neighbourhoods.csv")
#df_modelo = pd.read_csv(r"C:\Users\jaime\Documents\ICAI\Quinto\PrácticaFinal_BI_EstadisticaComp\AirbnbData\CleanData\df_modelo_limpio.csv")
#modelo = joblib.load(r"C:\Users\jaime\Documents\ICAI\Quinto\PrácticaFinal_BI_EstadisticaComp\Modelos\modelo_RF_baseline.pkl")

# Diego
df_listings = pd.read_csv(
    "/Users/diegoma/kaggle/df_listings_detailed_limpio.csv")
df_neighbourhoods = pd.read_csv("/Users/diegoma/kaggle/neighbourhoods.csv")
df_modelo = pd.read_csv("/Users/diegoma/kaggle/df_modelo_limpio.csv")
modelo = joblib.load("/Users/diegoma/kaggle/modelo_RF_baseline.pkl")
listings = pd.read_csv("/Users/diegoma/kaggle/listings_detailed.csv")
modelo_rev = joblib.load("/Users/diegoma/kaggle/modelo_RF_reviews.pkl")

tokenizer = AutoTokenizer.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment")
model_nlp = AutoModelForSequenceClassification.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment")
multilang_classifier = pipeline(
    "sentiment-analysis", model=model_nlp, tokenizer=tokenizer)

# --------------------------------------------------------------------
# Dashboard Ini
app = dash.Dash(__name__)

# --------------------------------------------------------------------
# Tool variables

count_neigh = len(df_neighbourhoods["neighbourhood_group"].unique())
unique_neigh = df_neighbourhoods["neighbourhood_group"].unique()
neigh_count = {}

options_neigh = []
for i in unique_neigh:
    options_neigh.append({'value': i, 'label': i})

unique_room = df_listings["room_type"].unique()
options_room = []
for i in unique_room:
    options_room.append({'value': i, 'label': i})


for i in range(count_neigh):
    neigh_count[unique_neigh[i]] = df_listings["neighbourhood_group_cleansed"][df_listings["neighbourhood_group_cleansed"]
                                                                               == unique_neigh[i]].count()

ordered_keys = []
ordered_values = []

for w in sorted(neigh_count, key=neigh_count.get):
    ordered_keys.append(w)
    ordered_values.append(neigh_count[w])

median_neigh = {}
for i in range(count_neigh):
    median_neigh[unique_neigh[i]] = round(
        df_listings[df_listings["neighbourhood_group_cleansed"] == unique_neigh[i]]["price"].median(), 3)

level_count = pd.DataFrame(df_listings["room_type"].value_counts()).reset_index(
).rename(columns={"index": "room_type", "room_type": "count"})
level_count = level_count.sort_values("room_type").reset_index(drop=True)

# --------------------------------------------------------------------
# Style variables
tab_style = {
    'font-family': 'verdana',
    'background-color': 'snow'
}
tab_selected_style = {
    'font-family': 'verdana'
}
style_texto = {
    'font-family': 'verdana',
    'margin-left': 80,
    'margin-right': 80
}
style_texto_2 = {
    'font-family': 'verdana',
    'size': 9
}
style_input = {
    'font-family': 'verdana',
    'margin-left': 80,
    'width': 390
}
dropdown_style = {
    'font-family': 'verdana',
    'padding-left': 80,
    'width': 500
}
dropdown_style_2 = {
    'font-family': 'verdana',
    'padding-left': 80,
    'width': 400
}
colorines = ["aliceblue", "antiquewhite", "aqua", "beige", "bisque", "black", "blueviolet", "cornsilk", "darkkhaki", "darkgrey", "darkgreen",
             "firebrick", "gainsboro", "ivory", "crimson", "darkslategray", "indigo", "mediumaquamarine", "olivedrab", "peachpuff", "mediumspringgreen"]

# --------------------------------------------------------------------
# App layout
app.layout = html.Div([

    html.Br(),
    html.H1(
        children="Dashboard Airbnb",
        style={
            'text-align': 'center',
            'font-family': 'verdana',
            "border-style": "outset",
            'height': 60,
            'padding-up': 10
        }
    ),

    html.Br(),

    dcc.Tabs([

        # Pestaña de resultados del modelo y app
        dcc.Tab(
            label='Predicción de precio',
            style=tab_style,
            selected_style=tab_selected_style,
            children=[
                html.Br(),
                html.Br(),
                html.H3(
                    children=[
                        "Prediccion de precio"
                    ],
                    id="subtituloModelo",
                    style={
                        "text-align": "center",
                        "font-family": "verdana",
                        "display": "block"
                    }
                ),
                html.P(
                    style={
                        "text-align": "center",
                        "font-family": "verdana",
                        "display": "block"
                    }
                ),
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.H5(
                                    "Es usted superhost:"
                                ),
                                html.Br(),
                                dcc.RadioItems(
                                    id="superhost",
                                    options=[{'label': 'Si', 'value': 'True'}, {
                                        'label': 'No', 'value': 'False'}],
                                    labelStyle={'display': 'inline-block'}
                                ),
                                html.Br(),
                                html.Br(),
                                html.Br(),
                                html.H5(
                                    "Escoja el barrio correspondiente: "
                                ),
                                html.Br(),
                                dcc.Dropdown(
                                    options=options_neigh,
                                    placeholder="Selecciona barrio",
                                    id="dropdown_neighb",
                                    value="",
                                    style={
                                        "display": "block",
                                        "width": "300px",
                                        "margin-left": "10px"
                                    }
                                ),
                            ],
                            style={
                                "width": "300px",
                                "height": "200px",
                                "display": "inline-block",
                                "margin": "30px"
                            }
                        ),
                        html.Div(
                            children=[
                                html.H5(
                                    "Escoja el tipo de habitación: "
                                ),
                                html.Br(),
                                dcc.Dropdown(
                                    options=options_room,
                                    placeholder="Selecciona el tipo de habitación",
                                    id="dropdown_room",
                                    style={
                                        "display": "block",
                                        "width": "300px",
                                        "margin-left": "10px"
                                    }
                                ),
                                html.Br(),
                                html.Br(),
                                html.Br(),
                                html.H5(
                                    "Introduzca el número de camas: "
                                ),
                                html.Br(),
                                dcc.Input(
                                    id="AccomInput",
                                    type="number",
                                    min=1,
                                    max=20
                                )
                            ],
                            style={
                                "width": "300px",
                                "height": "200px",
                                "display": "inline-block",
                                "margin": "30px"
                            }
                        ),
                        html.Div(
                            children=[
                                html.H5(
                                    "Introduzca el número de dormitorios: "
                                ),
                                html.Br(),
                                dcc.Input(
                                    id="BedroomsInput",
                                    type="number",
                                    min=1,
                                    max=20
                                ),
                                html.Br(),
                                html.Br(),
                                html.Br(),
                                html.H5(
                                    "Introduzca el número de baños: "
                                ),
                                html.Br(),
                                dcc.Input(
                                    id="BathInput",
                                    type="number",
                                    min=1,
                                    max=20
                                ),
                            ],
                            style={
                                "width": "300px",
                                "height": "200px",
                                "display": "inline-block",
                                "margin": "30px"
                            }
                        ),
                        html.Div(
                            children=[
                                html.H5(
                                    "Introduzca el número medio de reviews mensuales: "
                                ),
                                html.Br(),
                                dcc.Input(
                                    id="monthReviews",
                                    type="number",
                                    min=0,
                                    max=30
                                ),
                                html.Br(),
                                html.Br(),
                                html.Br(),
                                html.H5(
                                    "Introduzca el número de noches disponibles en un año: "
                                ),
                                html.Br(),
                                dcc.Input(
                                    id="Availab365",
                                    type="number",
                                    min=0,
                                    max=365
                                ),
                            ],
                            style={
                                "width": "300px",
                                "height": "200px",
                                "display": "inline-block",
                                "margin": "30px"
                            }
                        ),
                        html.Div(
                            children=[
                                html.H5(
                                    "Introduzca el número total de reviews: "
                                ),
                                html.Br(),
                                dcc.Input(
                                    id="numberReviews",
                                    type="number",
                                    min=1,
                                    max=800
                                ),
                                html.Br(),
                                html.Br(),
                                html.Br(),
                                html.H5(
                                    "Introduzca el número de publicaciones del host: "
                                ),
                                html.Br(),
                                dcc.Input(
                                    id="numberListings",
                                    type="number",
                                    min=1,
                                    max=300
                                ),
                            ],
                            style={
                                "height": "300px",
                                "display": "inline-block",
                                "margin": "30px"
                            }
                        )
                    ],
                    style={
                        "text-align": "center",
                        "border-style": "outset",
                        "border-color": "snow",
                        "border-width": "5px",
                        "background-color": "snow ",
                        'font-family': 'verdana',
                        "margin": "20px",
                        "width": "48%",
                        "display": "inline-block"
                    }
                ),

                html.Div(
                    children=[
                        dcc.Graph(
                            id='my-map'
                        )
                    ],
                    style={
                        "width": "48%",
                        "height": "700px",
                        "display": "inline-block",
                        "verticalAlign": "top"
                    }
                ),

                html.Div(
                    children=[
                        html.Br(),
                        html.Button('Enviar',
                                    id='submit-button',
                                    n_clicks=0,
                                    style={
                                        "border-radius": "15px",
                                        "cursor": "pointer",
                                        "padding": "15px 25px",
                                        "font-family": "verdana"
                                    }
                                    ),
                        html.Br(),
                        html.P(id='app-text-output',
                               style={

                               },
                               children='Texto Previo'
                               )
                    ],
                    style={
                        "text-align": "center"
                    }
                )
            ]
        ),

        # Análisis de texto
        dcc.Tab(label='Análisis de Texto', style=tab_style,
                selected_style=tab_selected_style, children=[
                    html.Div(
                        children=[
                            html.H3(
                                "Introduzca una review: "
                            ),
                            html.Br(),
                            dcc.Textarea(
                                id="review",
                                placeholder="Introduzca review",
                                value="",
                                style={'width': '500px',
                                       'height': "200px", "margin": "auto", "align": "center", }
                            ),
                            html.Br(),
                            html.Button('Analizar',
                                        id='submit-rev-button',
                                        n_clicks=0,
                                        style={
                                            "border-radius": "15px",
                                            "cursor": "pointer",
                                            "padding": "15px 25px",
                                            "font-family": "verdana"
                                        }
                                        )
                        ],
                        style={
                            "width": "300px",
                            "height": "300px",
                            "display": "inline-block",
                            "margin": "30px",
                            "align": "center",
                            "justify": "center"
                        }
                    ),
                    html.Div(
                        children=[
                            html.P(id='app-reviews-output',
                                   style={

                                   },
                                   children='Texto Previo'
                                   )
                        ],
                        style={
                            "width": "300px",
                            "height": "300px",
                            "display": "inline-block",
                            "margin": "30px",
                            "align": "center",
                            "justify": "center"
                        }
                    )
                ]
                ),

    ])

])

# -----------------------------------------------------------------------------------------
# Callback


@app.callback(
    Output(component_id='app-reviews-output', component_property='children'),
    Input(component_id='submit-rev-button', component_property='n_clicks'),
    Input(component_id='superhost', component_property='value'),
    Input(component_id='dropdown_neighb', component_property='value'),
    Input(component_id='dropdown_room', component_property='value'),
    Input(component_id='AccomInput', component_property='value'),
    Input(component_id='BedroomsInput', component_property='value'),
    Input(component_id='BathInput', component_property='value'),
    Input(component_id='monthReviews', component_property='value'),
    Input(component_id='Availab365', component_property='value'),
    Input(component_id='numberReviews', component_property='value'),
    Input(component_id='numberListings', component_property='value'),
    Input(component_id='review', component_property='value'),
)
def update_review(n_clicks, superhost, neighb, room, accomodates, beds, bath, monthly_rev, availab, reviews, listings, review):
    if(n_clicks > 0):
        puntuacion = float(multilang_classifier(review)
                           [0]['label'].split(' ')[0])
        df_sin_punt = getDataFrame(bool(superhost == 'True'), neighb, room, accomodates,
                                   beds, bath, monthly_rev, availab, reviews, listings)
        df_sin_punt.insert(9, "puntuacion_media", puntuacion)
        pred = modelo_rev.predict(df_sin_punt)

        numero_estrellas = round(puntuacion, 0)
        if(numero_estrellas == 1):
            return 'La review ha obtenido ⭐, y el precio esperado es de {} euros por noche.'.format(int(pred))
        elif(numero_estrellas == 2):
            return 'La review ha obtenido ⭐⭐, y el precio esperado es de {} euros por noche.'.format(int(pred))
        elif(numero_estrellas == 3):
            return 'La review ha obtenido ⭐⭐⭐, y el precio esperado es de {} euros por noche.'.format(int(pred))
        elif(numero_estrellas == 4):
            return 'La review ha obtenido ⭐⭐⭐⭐, y el precio esperado es de {} euros por noche.'.format(int(pred))
        elif(numero_estrellas == 5):
            return 'La review ha obtenido ⭐⭐⭐⭐⭐, y el precio esperado es de {} euros por noche.'.format(int(pred))
        else:
            return 'La review ha obtenido 0 estrellas'

    else:
        return ''


@app.callback(
    Output(component_id='app-text-output', component_property='children'),
    Input(component_id='submit-button', component_property='n_clicks'),
    Input(component_id='superhost', component_property='value'),
    Input(component_id='dropdown_neighb', component_property='value'),
    Input(component_id='dropdown_room', component_property='value'),
    Input(component_id='AccomInput', component_property='value'),
    Input(component_id='BedroomsInput', component_property='value'),
    Input(component_id='BathInput', component_property='value'),
    Input(component_id='monthReviews', component_property='value'),
    Input(component_id='Availab365', component_property='value'),
    Input(component_id='numberReviews', component_property='value'),
    Input(component_id='numberListings', component_property='value'),
)
def update_model(n_clicks, superhost, neighb, room, accomodates, beds, bath, monthly_rev, availab, reviews, listings):
    """Recibe los inputs del modelo, y devuelve el valor a predecir.

    Parameters:
    n_clicks(int) = Será un 1 cuando el usuario clickee el boton de enviar, 0 de lo contrario
    identidad_verif(str) = Valor que escoja el usuario 'true' si es verificado
    superhost(str) = Valor que escoja el usuario 'true' si es superhost, 'false' de lo contrario
    neighb(str) = Valor del barrio escogido
    room(str) = Valor del tipo de habitacion escogido
    beds(int) = Numero de dormitorios del inmueble
    beds(int) = Numero de camas del inmueble
    availab(int) = Numero de noches disponible en un año
    reviews(int) = Numero de reviews que tiene la publicacion
    listings(int) = Numero de publicaciones del host
    bath(int) = Numero de baños del inmueble
    monthly_rev (int) = average monthly reviews

    Returns:
    En caso de que se cliquee el boton, retornara el precio en una frase. En caso contrario, no devuelve nada
    """
    if(n_clicks > 0):
        print(bool(superhost == 'True'))
        dataf = getDataFrame(bool(superhost == 'True'), neighb, room, accomodates,
                             beds, bath, monthly_rev, availab, reviews, listings)
        prediction = modelo.predict(dataf)[0]
        return 'El precio esperado es de {} euros por noche'.format(
            int(prediction)
        )
    else:
        return ''


def getDataFrame(superhost, neighb, room, accomodates, beds, bath, monthly_rev, availab, reviews, listings):
    """Recibe los inputs del modelo, y devuelve un dataframe que se empleará para predecir el precio.

    Parameters:
    superhost(str) = Valor que escoja el usuario 'true' si es superhost, 'false' de lo contrario
    neighb(str) = Valor del barrio escogido
    room(str) = Valor del tipo de habitacion escogido
    accomodates(int) = Numero de camas del inmueble
    beds(int) = Numero de dormitorios del inmueble
    bath(int) = Numero de baños del inmueble
    reviews_month(int) = Numero de reviews mensuales
    availab(int) = Numero de noches disponible en un año
    reviews(int) = Numero de reviews que tiene la publicacion
    listings(int) = Numero de publicaciones del host
    review(str) = Review escrita por el usuario 

    Returns:
    Devuelve un dataframe con los tipos adecuados para el modelo, y el formato listo para predecir.  
    """

    #puntuacion = float(multilang_classifier(review)[0]['label'].split(' ')[0])

    dat = {
        'host_is_superhost': superhost, 'host_identity_verified': True, 'beds': accomodates, 'bathrooms': bath, 'bedrooms': beds,
        'reviews_per_month': monthly_rev, 'availability_365': availab, 'number_of_reviews': reviews, 'calculated_host_listings_count': listings,
        'neighbourhood_group_cleansed_Arganzuela': 0, 'neighbourhood_group_cleansed_Barajas': 0, 'neighbourhood_group_cleansed_Carabanchel': 0,
        'neighbourhood_group_cleansed_Centro': 0, 'neighbourhood_group_cleansed_Chamartín': 0, 'neighbourhood_group_cleansed_Chamberí': 0,
        'neighbourhood_group_cleansed_Ciudad Lineal': 0, 'neighbourhood_group_cleansed_Fuencarral - El Pardo': 0, 'neighbourhood_group_cleansed_Hortaleza': 0, 'neighbourhood_group_cleansed_Latina': 0,
        'neighbourhood_group_cleansed_Moncloa - Aravaca': 0, 'neighbourhood_group_cleansed_Moratalaz': 0, 'neighbourhood_group_cleansed_Puente de Vallecas': 0, 'neighbourhood_group_cleansed_Retiro': 0,
        'neighbourhood_group_cleansed_Salamanca': 0, 'neighbourhood_group_cleansed_San Blas - Canillejas': 0, 'neighbourhood_group_cleansed_Tetuán': 0,
        'neighbourhood_group_cleansed_Usera': 0, 'neighbourhood_group_cleansed_Vicálvaro': 0, 'neighbourhood_group_cleansed_Villa de Vallecas': 0,
        'neighbourhood_group_cleansed_Villaverde': 0, 'room_type_Entire home/apt': 0, 'room_type_Hotel room': 0, 'room_type_Private room': 0, 'room_type_Shared room': 0
    }
    neigh_key = 'neighbourhood_group_cleansed_' + neighb
    print(room)
    room_key = 'room_type_' + room
    dat[neigh_key] = 1
    dat[room_key] = 1

    dataf = pd.DataFrame(data=dat, index=[0])
    return dataf


@app.callback(
    Output(component_id='my-map', component_property='figure'),
    Input(component_id='dropdown_neighb', component_property='value')
)
def update_map(barrio):

    if(barrio == ""):
        df_tool = listings
    else:
        df_tool = listings[listings['neighbourhood_group_cleansed'] == barrio]

    fig = go.Figure()
    fig = px.scatter_mapbox(
        data_frame=df_tool,
        lat=df_tool['latitude'],
        lon=df_tool['longitude'],
        mapbox_style='carto-positron',
        color=df_tool['neighbourhood_group_cleansed'],
        center=dict(lat=40.43, lon=-3.68),
        zoom=11.5,
        height=800
    )

    return fig


# -----------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
