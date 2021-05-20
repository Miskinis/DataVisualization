import dash_table
import numpy as np
import pandas as pd
import plotly.express as px  # (version 4.7.0)

import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from scipy import spatial
from sklearn import preprocessing
from sklearn.decomposition import PCA

app = dash.Dash('Data Visualization')

df_lithuania = pd.read_csv("data/lithuania_total.csv")
df_latvia = pd.read_csv("data/latvia_total.csv")
df_estonia = pd.read_csv("data/estonia_total.csv")
df_poland = pd.read_csv("data/poland_total.csv")
df_germany = pd.read_csv("data/germany_total.csv")

countries = ['Lithuania', 'Latvia', 'Estonia', 'Poland', 'Germany']
countries_alpha = ['LTU', 'LVA', 'EST', 'POL', 'DEU']

country_options = [{'label': countries[0], 'value': countries[0]}]
for country in countries[1:]:
    country_options.append({'label': country, 'value': country})

years = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
year_options = [{'label': years[0], 'value': years[0]}]
for year in years[1:]:
    year_options.append({'label': year, 'value': year})

dff = {"Lithuania": df_lithuania,
       "Latvia": df_latvia,
       "Estonia": df_estonia,
       "Poland": df_poland,
       "Germany": df_germany}


def GetStats(df):
    df_stats = df.describe().round(decimals=3)
    df_stats.insert(0, 'Stats', df_stats.index)
    return df_stats


def GetTable(df, id):
    return dash_table.DataTable(
        id=id,
        columns=[{"name": i, "id": i}
                 for i in df.columns],
        data=df.to_dict('records'),
        style_cell=dict(textAlign='left'),
        style_header=dict(backgroundColor="paleturquoise"),
        style_data=dict(backgroundColor="lavender", whiteSpace='normal', height='auto'),
    )


@app.callback(
    Output(component_id='country_label', component_property='children'),
    Output(component_id='stats_table', component_property='children'),
    Output(component_id='country_table', component_property='children'),
    Input(component_id='slct_country', component_property='value')
)
def DisplayCountry(country):
    df_country = dff[country]
    df_stats = GetStats(df_country.drop('Year', axis=1))
    label = f"{country} Stats"
    stats_table = GetTable(df_stats, f"table_stats_{country}")
    country_table = GetTable(df_country, f"table_{country}")
    return label, stats_table, country_table


def SumOfManhattanDistances(x, y):
    return sum(abs(a - b) for a, b in zip(x, y))


def GetDistance(country):
    df_country = dff[country]
    min_max_scaler = preprocessing.MinMaxScaler()
    df_country_scaled = pd.DataFrame(
        data=min_max_scaler.fit_transform(df_country.drop('Year', axis=1)),
        columns=df_country.keys()[1:])

    distances = {}
    for key in df_country.keys()[2:]:
        distance = SumOfManhattanDistances(df_country_scaled['Death due to suicide'],
                                           df_country_scaled[key])
        distances[key] = distance

    df_distances = pd.DataFrame({'Attributes against suicide rates': df_country.keys()[2:]})
    df_distances['Manhattan'] = distances.values()

    distances = {}
    for key in df_country.keys()[2:]:
        distance = spatial.distance.euclidean(df_country_scaled['Death due to suicide'],
                                              df_country_scaled[key])
        distances[key] = distance
    df_distances['Euclidean'] = distances.values()

    distances = {}
    for key in df_country.keys()[2:]:
        distance = spatial.distance.cosine(df_country_scaled['Death due to suicide'],
                                           df_country_scaled[key])
        distances[key] = distance
    df_distances['Cosine'] = distances.values()

    return df_distances


@app.callback(
    Output(component_id='distance_table', component_property='children'),
    Input(component_id='slct_country', component_property='value')
)
def DisplayDistanceTable(country):
    return GetTable(GetDistance(country), f"table_distances_{country}")


@app.callback(
    Output(component_id='distance_table_graph', component_property='figure'),
    Input(component_id='slct_country', component_property='value')
)
def DisplayDistanceGraph(country):
    df_distance = GetDistance(country)
    indexes = df_distance.pop("Attributes against suicide rates")
    df_distance.index = indexes
    return px.bar(data_frame=df_distance,
                  barmode='group',
                  labels={'value': 'Distance'})


@app.callback(
    Output(component_id='total_attribute_boxplot', component_property='figure'),
    Input(component_id='slct_attribute', component_property='value')
)
def DisplayBoxplots(attribute):
    total_attributes = pd.DataFrame()
    for country in dff.keys():
        total_attributes[country] = dff[country][attribute]

    attribute_graph = px.box(total_attributes,
                             labels={"value": "Cases", "variable": "Country"},
                             title='Death due to suicide per 100k cases')
    return attribute_graph


@app.callback(
    Output(component_id='attribute_graph', component_property='figure'),
    Input(component_id='slct_attribute', component_property='value')
)
def DisplayAttributesGraph(attribute):
    total_attribute = pd.DataFrame()
    for localCountry in dff.keys():
        total_attribute[localCountry] = dff[localCountry][attribute]
    total_attribute.index = dff[countries[0]]['Year'].values
    attribute_graph = px.line(total_attribute,
                              title=attribute,
                              labels={"x": "Year", "y": "Value"},
                              )

    return attribute_graph


def GetCorrelation(df, country):
    data_corr = df.corr().take([0], axis=1).take(range(1, len(df) - 1))
    data_corr.rename(columns={'Death due to suicide': country}, inplace=True)
    return pd.DataFrame(data_corr)


df_attributes = pd.DataFrame([
    {"Attribute": "Death due to suicide",
     "Type": "Discrete",
     "Unit Of Measure": "Rate",
     "Description": "Death rate of a population adjusted to a standard age distribution. As most causes of death vary significantly with people's age and sex, the use of standardised death rates improves comparability over time and between countries, as they aim at measuring death rates independently of different age structures of populations. The standardised death rates used here are calculated on the basis of a standard European population (defined by the World Health Organization)"
     },

    {"Attribute": "Tax rate on low wage earners",
     "Type": "Continuous",
     "Unit Of Measure": "Percentage",
     "Description": "Tax wedge on labour costs"
     },

    {"Attribute": "Real GDP growth rate",
     "Type": "Continuous",
     "Unit Of Measure": "Chain linked volumes, percentage change on previous period",
     "Description": "Gross domestic product (GDP) is a measure of the economic activity, defined as the value of all goods and services produced less the value of any goods or services used in their creation. The calculation of the annual growth rate of GDP volume is intended to allow comparisons of the dynamics of economic development both over time and between economies of different sizes. For measuring the growth rate of GDP in terms of volumes, the GDP at current prices are valued in the prices of the previous year and the thus computed volume changes are imposed on the level of a reference year; this is called a chain-linked series. Accordingly, price movements will not inflate the growth rate."
     },

    {"Attribute": "Long working hours in main job",
     "Type": "Continuous",
     "Unit Of Measure": "Percentage",
     "Description": "Long working hours in main job from age 15 to 64 years"
     },

    {"Attribute": "At-risk-of-poverty rate by poverty threshold",
     "Type": "Continuous",
     "Unit Of Measure": "Percentage",
     "Description": "At risk of poverty rate (cut-off point: 60% of median equivalised income after social transfers)"
     },

    {"Attribute": "Employment rate of adults",
     "Type": "Continuous",
     "Unit Of Measure": "Percentage",
     "Description": "Employment rate of adults with education (All ISCED 2011 levels), age of child less than 6 years, Age group from 20 to 49 years"
     }
])


def GetCorrelationGraph():
    total_corr = {}
    for country in dff.keys():
        total_corr[country] = GetCorrelation(dff[country].drop('Year', axis=1), country)

    total_corr = pd.concat(total_corr.values(), axis=1)

    # Plotly Express
    return px.bar(data_frame=total_corr,
                  title="Correlation with Deaths due to suicide",
                  barmode='group',
                  range_y=[-1, 1])


@app.callback(
    Output(component_id='suicide_rate_map', component_property='figure'),
    Input(component_id='slct_year', component_property='value')
)
def DisplaySuicideMap(year):
    suicide_rates = []
    current_year = []
    for country in countries:
        df_country = dff[country]
        year_index = [df_country.index[df_country['Year'] == year].tolist()]
        suicide_rates.append(df_country["Death due to suicide"][year_index[0]].values[0])
        current_year.append(year)

    df_suicide = pd.DataFrame({"Country": countries,
                               "iso_alpha": countries_alpha,
                               "Year": current_year,
                               "Suicide Rate": suicide_rates})

    fig = px.choropleth(df_suicide,
                        locations="iso_alpha",
                        color="Suicide Rate",
                        hover_name="Country",
                        color_continuous_scale="Turbo")
    fig.update_geos(fitbounds="locations", visible=True)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


@app.callback(
    Output(component_id='parallel_coordinates', component_property='figure'),
    Input(component_id='slct_country', component_property='value')
)
def DisplayParallelCoordinates(country):
    fig = px.parallel_coordinates(dff[country].drop("Year", axis=1),
                                  color="Death due to suicide",
                                  color_continuous_scale="Turbo")
    return fig


@app.callback(
    Output(component_id='pca_scatter_matrix', component_property='figure'),
    Input(component_id='slct_country', component_property='value'),
    Input(component_id='pca_components_slider', component_property='value')
)
def DisplayPcaScatterMatrix(country, slider_value):
    pca = PCA()
    features = dff[country].keys()[1:][1:]
    components = pca.fit_transform(dff[country][features])
    labels = {
        str(i): f"PC {i + 1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }
    labels['color'] = 'Death due to suicide'

    fig = px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(slider_value),
        color=dff[country]["Death due to suicide"]
    )
    fig.update_traces(diagonal_visible=False)
    return fig


@app.callback(
    Output(component_id='pca_scatter_3D', component_property='figure'),
    Input(component_id='slct_country', component_property='value'),
)
def DisplayPcaScatter3D(country):
    pca = PCA(n_components=3)
    features = dff[country].keys()[1:][1:]
    components = pca.fit_transform(dff[country][features])
    total_var = pca.explained_variance_ratio_.sum() * 100
    fig = px.scatter_3d(
        components, x=0, y=1, z=2, color=dff[country]['Death due to suicide'],
        title=f'Total Explained Variance: {total_var:.2f}%',
        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3', 'color': 'Death due to suicide'}
    )
    return fig


@app.callback(
    Output(component_id='pca_area', component_property='figure'),
    Input(component_id='slct_country', component_property='value'),
)
def DisplayVarianceArea(country):
    features = dff[country].keys()[1:][1:]
    df = dff[country][features]
    pca = PCA()
    pca.fit(df)
    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

    return px.area(
        x=range(1, exp_var_cumul.shape[0] + 1),
        y=exp_var_cumul,
        labels={"x": "# Components", "y": "Explained Variance"}
    )

@app.callback(
    Output(component_id='pca_variance_loadings', component_property='figure'),
    Input(component_id='slct_country', component_property='value'),
)
def DisplayVarianceLoadings(country):
    df = dff[country]
    features = dff[country].keys()[1:][1:]
    X = df[features]

    pca = PCA()
    components = pca.fit_transform(X)

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    fig = px.scatter(components, x=0, y=1, color=df['Death due to suicide'], labels={'color': 'Death due to suicide'})

    for i, feature in enumerate(features.values):
        fig.add_shape(
            type='line',
            x0=0, y0=0,
            x1=loadings[i, 0],
            y1=loadings[i, 1]
        )
        fig.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 1],
            ax=0, ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
        )
    return fig

# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([

    html.H1("Are selected factors related to death rates in certain countries (statistical data)",
            style={'text-align': 'center'}),
    html.Br(),

    html.Div(id='attributes_table_graph', children=GetTable(df_attributes, 'attributes_table')),
    html.Br(),

    dcc.Dropdown(id="slct_year",
                 options=year_options,
                 multi=False,
                 value=2011,
                 style={'width': "40%"}
                 ),

    dcc.Graph(id='suicide_rate_map'),

    html.Br(),
    html.H3("Certain countries factors correlation with correlation with deaths due to suicide",
            style={'text-align': 'center'}),
    html.Br(),
    dcc.Graph(id='corr_with_death_rates', figure=GetCorrelationGraph()),

    dcc.Dropdown(id="slct_attribute",
                 options=[
                     {"label": "Death due to suicide", "value": "Death due to suicide"},
                     {"label": "Tax rate on low wage earners", "value": "Tax rate on low wage earners"},
                     {"label": "Real GDP growth rate", "value": "Real GDP growth rate"},
                     {"label": "Long working hours in main job", "value": "Long working hours in main job"},
                     {"label": "At-risk-of-poverty rate by poverty threshold",
                      "value": "At-risk-of-poverty rate by poverty threshold"},
                     {"label": "Employment rate of adults", "value": "Employment rate of adults"}],
                 multi=False,
                 value="Death due to suicide",
                 style={'width': "40%"}
                 ),

    html.Br(),
    dcc.Graph(id='total_attribute_boxplot', figure={}),
    html.Br(),
    dcc.Graph(id='attribute_graph', figure={}),
    html.Br(),

    dcc.Dropdown(id="slct_country",
                 options=country_options,
                 multi=False,
                 value="Lithuania",
                 style={'width': "40%"}
                 ),

    html.Br(),
    html.H2(id="country_label", style={'text-align': 'center'}),
    html.Br(),

    html.Div(id='stats_table', children=[]),
    html.Br(),

    html.Div(id='country_table', children=[]),
    dcc.Graph(id='parallel_coordinates'),
    html.Br(),

    html.H3("Calculated distance between Death due to suicide and all other attributes, by using 3 different methods",
            style={'text-align': 'center'}),
    html.Br(),
    html.Div(id='distance_table', children=[]),
    html.Br(),
    dcc.Graph(id='distance_table_graph'),
    html.Br(),
    html.H3("Subsets of Principal Components", style={'text-align': 'center'}),
    html.Div([
        html.P("Number of components:"),
        dcc.Slider(
            id='pca_components_slider',
            min=2, max=5, value=3,
            marks={i: str(i) for i in range(2, 6)}),
        dcc.Graph(id='pca_scatter_matrix'),
    ]),
    html.Br(),
    html.H3("Variance Loadings", style={'text-align': 'center'}),
    dcc.Graph(id='pca_variance_loadings'),
    html.Br(),
    html.H3("Subset of 3 Principal Components", style={'text-align': 'center'}),
    dcc.Graph(id='pca_scatter_3D'),
    html.Br(),
    html.H3("Variance based on component count", style={'text-align': 'center'}),
    dcc.Graph(id='pca_area'),
])

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
