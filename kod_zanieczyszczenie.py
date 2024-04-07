import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.geocoders import Nominatim
from tqdm import tqdm
import plotly.express as px
import ipywidgets as widgets
from itables import init_notebook_mode
from itables import show

data = pd.read_excel('C:/ML_Prezentacja/projekt_Zanieczyszczenia/data.xlsx',sheet_name='AAP_2022_city_v9')
print(len(data))

#data_test = pd.read_excel('C:/ML_Prezentacja/projekt_Zanieczyszczenia/data.xlsx', sheet_name='AAP_2022_city_v9')
#show(data_test)

# Wybór wierszy, gdzie PM2.5 i PM10 mają puste wartości
missing_values_rows = data[data['PM2.5 (μg/m3)'].isnull() & data['PM10 (μg/m3)'].isnull()]
print("Liczba wierszy, gdzie PM2.5 i PM10 mają puste wartości:", len(missing_values_rows))

# Usunięcię wierszy z pustymi rekordami zarówno dla PM2.5 jak i dla PM10
data = data.dropna(subset=['PM2.5 (μg/m3)', 'PM10 (μg/m3)'], how='all')
# Usunięcie danych z lat poniżej 2010 roku, ponieważ dla jednego z krajów podawane są dane z lat wcześniejszych
data = data[data['Measurement Year'] >= 2010]
print(data[['City or Locality', 'PM2.5 (μg/m3)', 'PM10 (μg/m3)']].head(30))

# Obliczenie różnic między PM10 a PM2.5 dla każdego miasta
difference_by_city = data.groupby('City or Locality')[['PM10 (μg/m3)', 'PM2.5 (μg/m3)']].apply(lambda x: x['PM10 (μg/m3)'] - x['PM2.5 (μg/m3)'])

# Uzupełnienie brakujących wartości na podstawie różnicy
data['PM2.5 (μg/m3)'] = data.apply(
    lambda row: round(row['PM10 (μg/m3)'] - difference_by_city[row['City or Locality']].mean(), 2) if pd.isna(
        row['PM2.5 (μg/m3)']) else row['PM2.5 (μg/m3)'], axis=1)
data['PM10 (μg/m3)'] = data.apply(
    lambda row: round(row['PM2.5 (μg/m3)'] + difference_by_city[row['City or Locality']].mean(), 2) if pd.isna(
        row['PM10 (μg/m3)']) else row['PM10 (μg/m3)'], axis=1)

print(data[['City or Locality', 'PM2.5 (μg/m3)', 'PM10 (μg/m3)']].head(30))

data = data[(data['PM2.5 (μg/m3)'] >= 2)]
print(len(data))

data2 = pd.read_excel('C:/ML_Prezentacja/projekt_Zanieczyszczenia/cities_names.xlsx', engine='openpyxl')

# Wyodrębnienie unikalnych miast z obu plików z danymi
cities1 = set(data['City or Locality'].unique())
cities2 = set(data2['ASCII Name'].unique())

# Znalezienie wspólnych miast w obu plikach
common_cities = cities1.intersection(cities2)

# Filtracja danych z pliku2 dla miast wspólnych
filtered_data2 = data2[data2['ASCII Name'].isin(common_cities)]

# Połączenie dane1 i dane2 na podstawie wspólnych miast, dodając kolumny z dane2
merged_data = pd.merge(data, filtered_data2[['ASCII Name', 'Coordinates']], left_on='City or Locality', right_on='ASCII Name', suffixes=('_data', '_data1'), how='left')

merged_data.to_excel('C:/ML_Prezentacja/projekt_Zanieczyszczenia/cities_names_coordinates.xlsx', index=False)

# Sprawdzenie ile zostało nieuzupełnionych danych
empty_coordinates_data = merged_data[merged_data['Coordinates'].isna()].copy()
empty_coordinates_count = merged_data['Coordinates'].isna().sum()
print("Liczba pustych rekordów w tabeli Coordinates:", empty_coordinates_count)

# Stworzenie nowego pliku jedynie z danymi bez koordynatów, bez powtarzania miast
data_unique_cities = empty_coordinates_data.drop_duplicates(subset='City or Locality', keep='first')
data_unique_cities.to_excel('C:/ML_Prezentacja/projekt_Zanieczyszczenia/unique_cities.xlsx', index=False)

# Funkcja do pobierania koordynatów dla danego miasta
def get_coordinates(city):
    try:
        geolocator = Nominatim(user_agent="geoapiExercises")
        location = geolocator.geocode(city)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except:
        return None, None


# Utworzony został pasek postępu aby wiedzieć ile zostało do zakończenia kompilacji
# pbar = tqdm(total=len(data_unique_cities), desc="Updating coordinates")
# # Aktualizacja wartości kolumny Coordinates, wykorzystując funkcję get_coordinates
# for index, row in data_unique_cities.iterrows():
#     city = row['City or Locality']
#     latitude, longitude = get_coordinates(city)
#     data_unique_cities.at[index, 'Latitude'] = latitude
#     data_unique_cities.at[index, 'Longitude'] = longitude
#     pbar.update(1)

# pbar.close()

# data_unique_cities.to_excel('C:/ML_Prezentacja/projekt_Zanieczyszczenia/unique_cities_with_coordinates.xlsx',index=False)

cities_coordinates = pd.read_excel('C:/ML_Prezentacja/projekt_Zanieczyszczenia/cities_names_coordinates.xlsx')
cities_coordinates[['Latitude', 'Longitude']] = cities_coordinates['Coordinates'].str.split(',', expand=True)
#Usunięcie niepotrzebnej już kolumny Coordinates
cities_coordinates.drop(columns=['Coordinates'], inplace=True)

cities_coordinates.to_excel('C:/ML_Prezentacja/projekt_Zanieczyszczenia/cities_names_coordinates_updated.xlsx', index=False)
cities_names_coordinates = pd.read_excel('C:/ML_Prezentacja/projekt_Zanieczyszczenia/cities_names_coordinates_updated.xlsx')
unique_cities_with_coordinates = pd.read_excel('C:/ML_Prezentacja/projekt_Zanieczyszczenia/unique_cities_with_coordinates.xlsx')

data = pd.merge(data, cities_names_coordinates[['City or Locality', 'Longitude', 'Latitude']], on='City or Locality', how='left')
data= pd.merge(data, unique_cities_with_coordinates[['City or Locality', 'Longitude', 'Latitude']], on='City or Locality', how='left')
# Wybranie kolumny Longitude i Latitude
data['Longitude'] = data['Longitude_x'].fillna(data['Longitude_y'])
data['Latitude'] = data['Latitude_x'].fillna(data['Latitude_y'])
# Usunięcie zbędnych kolumn pomocniczych
data.drop(['Longitude_x', 'Latitude_x', 'Longitude_y', 'Latitude_y'], axis=1, inplace=True)
data.drop_duplicates(subset=['City or Locality', 'Measurement Year'], inplace=True)
data.loc[data['City or Locality'] == 'Adelaide', ['Latitude', 'Longitude']] = [-34.921230, 138.599503]
# Zapisywanie danych do pliku
data.to_excel('C:/ML_Prezentacja/projekt_Zanieczyszczenia/data_ready.xlsx', index=False)

dane = pd.read_excel('C:/ML_Prezentacja/projekt_Zanieczyszczenia/data_ready.xlsx')

countries_in_regions = {}

# Grupowanie danych po regionie i unikalnych krajach
region_country_data = dane.groupby('WHO Region')['WHO Country Name'].unique()

for region, countries in region_country_data.items():
    countries_in_regions[region] = countries

# Wyświetlenie listy krajów wchodzących w poszczególne regiony
for region, countries in countries_in_regions.items():
    print(f"{region}: {', '.join(countries)}\n")


# Funkcja do aktualizacji mapy
def update_map(selected_pollutant):
    # Filtracja danych dla regionu Europy
    europe_data = dane[dane['WHO Region'] == 'European Region']

    # Usunięcie wierszy, w których wybrany czynnik jest null
    europe_data_filtered = europe_data.dropna(subset=[selected_pollutant])

    # Sortowanie danych po roku
    europe_data_sorted = europe_data_filtered.sort_values(by='Measurement Year')

    # Tworzenie interaktywnej mapy z jednakową skalą wartości dla wszystkich lat
    fig = px.scatter_mapbox(europe_data_sorted, lat='Latitude', lon='Longitude', color=selected_pollutant,
                            hover_name='City or Locality', hover_data=[selected_pollutant, 'PM10 (μg/m3)', 'NO2 (μg/m3)'],
                            animation_frame='Measurement Year',
                            title=f'Zanieczyszczenie powietrza w regionie Europy na przestrzeni lat 2010-2020 ({selected_pollutant})',
                            mapbox_style='carto-positron', zoom=3, height=750,
                            range_color=[europe_data_sorted[selected_pollutant].min(), europe_data_sorted[selected_pollutant].max()],
                            center={'lat': europe_data_sorted['Latitude'].mean(), 'lon': europe_data_sorted['Longitude'].mean()})
    # Wyświetlenie mapy
    fig.show()

# Tworzenie interaktywnego select boxa dla czynnika zanieczyszczenia
select_pollutant = widgets.Dropdown(options=['PM2.5 (μg/m3)', 'PM10 (μg/m3)', 'NO2 (μg/m3)'], description='Czynnik:', style={'description_width': 'initial'})

# Wywołanie funkcji update_map po zmianie wartości select boxa
widgets.interactive(update_map, selected_pollutant=select_pollutant)


# Funkcja do aktualizacji mapy po wybraniu kraju, roku i czynnika
def update_map(selected_pollutant, selected_country, selected_year):
    # Filtracja danych dla wybranego kraju, roku i czynnika
    country_year_pollutant_data = dane[(dane['WHO Country Name'] == selected_country) &
                                       (dane['Measurement Year'] == selected_year) &
                                       (dane[selected_pollutant].notna())]

    # Tworzenie interaktywnej mapy dla wybranego kraju, roku i czynnika
    fig = px.scatter_mapbox(country_year_pollutant_data, lat='Latitude', lon='Longitude', color=selected_pollutant,
                            hover_name='City or Locality',
                            hover_data=[selected_pollutant, 'PM10 (μg/m3)', 'NO2 (μg/m3)'],
                            title=f'Zanieczyszczenie powietrza w kraju: {selected_country}, rok: {selected_year} ({selected_pollutant})',
                            mapbox_style='carto-positron', zoom=3, height=600,
                            range_color=[country_year_pollutant_data[selected_pollutant].min(),
                                         country_year_pollutant_data[selected_pollutant].max()],
                            center={'lat': country_year_pollutant_data['Latitude'].mean(),
                                    'lon': country_year_pollutant_data['Longitude'].mean()})

    # Wyświetlenie mapy
    fig.show()


# Funkcja do aktualizacji dostępnych czynników zanieczyszczenia dla wybranego kraju
def update_pollutants_options(selected_country):
    available_pollutants = ['PM2.5 (μg/m3)', 'PM10 (μg/m3)', 'NO2 (μg/m3)']
    select_pollutant.options = available_pollutants

    # Sprawdzenie czy domyślna wartość jest obecna na liście dostępnych czynników zanieczyszczenia dla wybranego kraju
    if select_pollutant.value not in available_pollutants:
        select_pollutant.value = available_pollutants[
            0]  # Ustawienie pierwszego dostępnego czynnika jako domyślną wartość

    # Aktualizacja dostępnych krajów na podstawie wybranego czynnika
    available_countries = dane[dane[selected_pollutant].notna()]['WHO Country Name'].unique()
    select_country.options = available_countries

    # Sprawdzenie czy wybrany kraj jest obecny na liście dostępnych krajów dla wybranego czynnika
    if select_country.value not in available_countries:
        select_country.value = available_countries[0]  # Ustawienie pierwszego dostępnego kraju jako domyślną wartość


# Funkcja do aktualizacji dostępnych lat dla wybranego kraju i czynnika
def update_years_options(selected_country, selected_pollutant):
    available_years = sorted(dane[(dane['WHO Country Name'] == selected_country) &
                                  (dane[selected_pollutant].notna())]['Measurement Year'].unique())
    select_year.options = available_years

    # Sprawdzenie czy wybrany rok jest obecny na liście dostępnych lat dla wybranego kraju i czynnika
    if select_year.value not in available_years:
        select_year.value = available_years[0]  # Ustawienie pierwszego dostępnego roku jako domyślną wartość


# Lista dostępnych krajów w Region of the Americas
european_countries = dane[dane['WHO Region'] == 'European Region']['WHO Country Name'].unique()

# Tworzenie interaktywnych select boxów
select_pollutant = widgets.Dropdown(description='Czynnik:', style={'description_width': 'initial'})
select_country = widgets.Dropdown(options=european_countries, description='Kraj:',
                                  style={'description_width': 'initial'})
select_year = widgets.Dropdown(description='Rok:', style={'description_width': 'initial'})

widgets.interactive(update_years_options, selected_country=select_country, selected_pollutant=select_pollutant)
widgets.interactive(update_pollutants_options, selected_country=select_country)

widgets.interactive(update_map, selected_pollutant=select_pollutant, selected_country=select_country,
                    selected_year=select_year)

# Funkcja do aktualizacji mapy
def update_map(selected_pollutant):
    # Filtracja danych dla regionu Mediterranean
    mediterranean_data = dane[dane['WHO Region'] == 'Eastern Mediterranean Region']

    # Usunięcie wierszy, w których wybrany czynnik jest null
    mediterranean_data_filtered = mediterranean_data.dropna(subset=[selected_pollutant])

    # Sortowanie danych po roku
    mediterranean_data_sorted = mediterranean_data_filtered.sort_values(by='Measurement Year')

    # Tworzenie interaktywnej mapy z jednakową skalą wartości dla wszystkich lat
    fig = px.scatter_mapbox(mediterranean_data_sorted, lat='Latitude', lon='Longitude', color=selected_pollutant,
                            hover_name='City or Locality', hover_data=[selected_pollutant, 'PM10 (μg/m3)', 'NO2 (μg/m3)'],
                            animation_frame='Measurement Year',
                            title=f'Zanieczyszczenie powietrza w regionie Mediterranean na przestrzeni lat 2010-2020 ({selected_pollutant})',
                            mapbox_style='carto-positron', zoom=3, height=750,
                            range_color=[mediterranean_data_sorted[selected_pollutant].min(), mediterranean_data_sorted[selected_pollutant].max()],
                            center={'lat': mediterranean_data_sorted['Latitude'].mean(), 'lon': mediterranean_data_sorted['Longitude'].mean()})
    # Wyświetlenie mapy
    fig.show()

# Tworzenie interaktywnego select boxa dla czynnika zanieczyszczenia
select_pollutant = widgets.Dropdown(options=['PM2.5 (μg/m3)', 'PM10 (μg/m3)', 'NO2 (μg/m3)'], description='Czynnik:', style={'description_width': 'initial'})

# Wywołanie funkcji update_map po zmianie wartości select boxa
widgets.interactive(update_map, selected_pollutant=select_pollutant)


# Funkcja do aktualizacji mapy po wybraniu kraju, roku i czynnika
def update_map(selected_pollutant, selected_country, selected_year):
    # Filtracja danych dla wybranego kraju, roku i czynnika
    country_year_pollutant_data = dane[(dane['WHO Country Name'] == selected_country) &
                                       (dane['Measurement Year'] == selected_year) &
                                       (dane[selected_pollutant].notna())]

    # Tworzenie interaktywnej mapy dla wybranego kraju, roku i czynnika
    fig = px.scatter_mapbox(country_year_pollutant_data, lat='Latitude', lon='Longitude', color=selected_pollutant,
                            hover_name='City or Locality',
                            hover_data=[selected_pollutant, 'PM10 (μg/m3)', 'NO2 (μg/m3)'],
                            title=f'Zanieczyszczenie powietrza w kraju: {selected_country}, rok: {selected_year} ({selected_pollutant})',
                            mapbox_style='carto-positron', zoom=3, height=600,
                            range_color=[country_year_pollutant_data[selected_pollutant].min(),
                                         country_year_pollutant_data[selected_pollutant].max()],
                            center={'lat': country_year_pollutant_data['Latitude'].mean(),
                                    'lon': country_year_pollutant_data['Longitude'].mean()})

    # Wyświetlenie mapy
    fig.show()


# Funkcja do aktualizacji dostępnych czynników zanieczyszczenia dla wybranego kraju
def update_pollutants_options(selected_country):
    available_pollutants = ['PM2.5 (μg/m3)', 'PM10 (μg/m3)', 'NO2 (μg/m3)']
    select_pollutant.options = available_pollutants

    # Sprawdzenie czy domyślna wartość jest obecna na liście dostępnych czynników zanieczyszczenia dla wybranego kraju
    if select_pollutant.value not in available_pollutants:
        select_pollutant.value = available_pollutants[
            0]  # Ustawienie pierwszego dostępnego czynnika jako domyślną wartość

    # Aktualizacja dostępnych krajów na podstawie wybranego czynnika
    available_countries = dane[dane[selected_pollutant].notna()]['WHO Country Name'].unique()
    select_country.options = available_countries

    # Sprawdzenie czy wybrany kraj jest obecny na liście dostępnych krajów dla wybranego czynnika
    if select_country.value not in available_countries:
        select_country.value = available_countries[0]  # Ustawienie pierwszego dostępnego kraju jako domyślną wartość


# Funkcja do aktualizacji dostępnych lat dla wybranego kraju i czynnika
def update_years_options(selected_country, selected_pollutant):
    available_years = sorted(dane[(dane['WHO Country Name'] == selected_country) &
                                  (dane[selected_pollutant].notna())]['Measurement Year'].unique())
    select_year.options = available_years

    # Sprawdzenie czy wybrany rok jest obecny na liście dostępnych lat dla wybranego kraju i czynnika
    if select_year.value not in available_years:
        select_year.value = available_years[0]  # Ustawienie pierwszego dostępnego roku jako domyślną wartość


# Lista dostępnych krajów w Region of the Americas
arabic_countries = dane[dane['WHO Region'] == 'Eastern Mediterranean Region']['WHO Country Name'].unique()

# Tworzenie interaktywnego select boxów
select_pollutant = widgets.Dropdown(description='Czynnik:', style={'description_width': 'initial'})
select_country = widgets.Dropdown(options=arabic_countries, description='Kraj:', style={'description_width': 'initial'})
select_year = widgets.Dropdown(description='Rok:', style={'description_width': 'initial'})

# Po wyborze kraju, aktualizuj dostępne lata i czynniki zanieczyszczenia
widgets.interactive(update_years_options, selected_country=select_country, selected_pollutant=select_pollutant)
widgets.interactive(update_pollutants_options, selected_country=select_country)

# Wywołanie funkcji update_map po zmianie wartości select boxów
widgets.interactive(update_map, selected_pollutant=select_pollutant, selected_country=select_country,
                    selected_year=select_year)

def update_map(selected_pollutant):
    african_data = dane[dane['WHO Region'] == 'African Region']
    african_data_filtered = african_data.dropna(subset=[selected_pollutant])
    african_data_sorted = african_data_filtered.sort_values(by='Measurement Year')

    fig = px.scatter_mapbox(african_data_sorted, lat='Latitude', lon='Longitude', color=selected_pollutant,
                            hover_name='City or Locality', hover_data=[selected_pollutant, 'PM10 (μg/m3)', 'NO2 (μg/m3)'],
                            animation_frame='Measurement Year',
                            title=f'Zanieczyszczenie powietrza w regionie African Region na przestrzeni lat 2010-2020 ({selected_pollutant})',
                            mapbox_style='carto-positron', zoom=3, height=750,
                            range_color=[african_data_sorted[selected_pollutant].min(), african_data_sorted[selected_pollutant].max()],
                            center={'lat': african_data_sorted['Latitude'].mean(), 'lon': african_data_sorted['Longitude'].mean()})
    fig.show()

select_pollutant = widgets.Dropdown(options=['PM2.5 (μg/m3)', 'PM10 (μg/m3)', 'NO2 (μg/m3)'], description='Czynnik:', style={'description_width': 'initial'})
widgets.interactive(update_map, selected_pollutant=select_pollutant)


def update_map(selected_pollutant, selected_country, selected_year):
    country_year_pollutant_data = dane[(dane['WHO Country Name'] == selected_country) &
                                       (dane['Measurement Year'] == selected_year) &
                                       (dane[selected_pollutant].notna())]

    fig = px.scatter_mapbox(country_year_pollutant_data, lat='Latitude', lon='Longitude', color=selected_pollutant,
                            hover_name='City or Locality',
                            hover_data=[selected_pollutant, 'PM10 (μg/m3)', 'NO2 (μg/m3)'],
                            title=f'Zanieczyszczenie powietrza w kraju: {selected_country}, rok: {selected_year} ({selected_pollutant})',
                            mapbox_style='carto-positron', zoom=3, height=600,
                            range_color=[country_year_pollutant_data[selected_pollutant].min(),
                                         country_year_pollutant_data[selected_pollutant].max()],
                            center={'lat': country_year_pollutant_data['Latitude'].mean(),
                                    'lon': country_year_pollutant_data['Longitude'].mean()})

    fig.show()


def update_pollutants_options(selected_country):
    available_pollutants = ['PM2.5 (μg/m3)', 'PM10 (μg/m3)', 'NO2 (μg/m3)']
    select_pollutant.options = available_pollutants

    if select_pollutant.value not in available_pollutants:
        select_pollutant.value = available_pollutants[0]

    available_countries = dane[dane[selected_pollutant].notna()]['WHO Country Name'].unique()
    select_country.options = available_countries

    if select_country.value not in available_countries:
        select_country.value = available_countries[0]


def update_years_options(selected_country, selected_pollutant):
    available_years = sorted(dane[(dane['WHO Country Name'] == selected_country) &
                                  (dane[selected_pollutant].notna())]['Measurement Year'].unique())
    select_year.options = available_years

    if select_year.value not in available_years:
        select_year.value = available_years[0]


african_countries = dane[dane['WHO Region'] == 'African Region']['WHO Country Name'].unique()

select_pollutant = widgets.Dropdown(description='Czynnik:', style={'description_width': 'initial'})

select_country = widgets.Dropdown(options=african_countries, description='Kraj:',
                                  style={'description_width': 'initial'})

select_year = widgets.Dropdown(description='Rok:', style={'description_width': 'initial'})

widgets.interactive(update_years_options, selected_country=select_country, selected_pollutant=select_pollutant)
widgets.interactive(update_pollutants_options, selected_country=select_country)

widgets.interactive(update_map, selected_pollutant=select_pollutant, selected_country=select_country,
                    selected_year=select_year)


def update_map(selected_pollutant):
    americas_data = dane[dane['WHO Region'].str.contains('Region of the Americas')]
    americas_data_filtered = americas_data.dropna(subset=[selected_pollutant])
    americas_data_sorted = americas_data_filtered.sort_values(by='Measurement Year')

    fig = px.scatter_mapbox(americas_data_sorted, lat='Latitude', lon='Longitude', color=selected_pollutant,
                            hover_name='City or Locality', hover_data=[selected_pollutant, 'PM10 (μg/m3)', 'NO2 (μg/m3)'],
                            animation_frame='Measurement Year',
                            title=f'Zanieczyszczenie powietrza w regionie Ameryki Północnej i Południowej na przestrzeni lat 2010-2020 ({selected_pollutant})',
                            mapbox_style='carto-positron', zoom=3, height=750,
                            range_color=[americas_data_sorted[selected_pollutant].min(), americas_data_sorted[selected_pollutant].max()],
                            center={'lat': americas_data_sorted['Latitude'].mean(), 'lon': americas_data_sorted['Longitude'].mean()})
    fig.show()

select_pollutant = widgets.Dropdown(options=['PM2.5 (μg/m3)', 'PM10 (μg/m3)', 'NO2 (μg/m3)'], description='Czynnik:', style={'description_width': 'initial'})

widgets.interactive(update_map, selected_pollutant=select_pollutant)


def update_map(selected_pollutant, selected_country, selected_year):
    country_year_pollutant_data = dane[(dane['WHO Country Name'] == selected_country) &
                                       (dane['Measurement Year'] == selected_year) &
                                       (dane[selected_pollutant].notna())]

    fig = px.scatter_mapbox(country_year_pollutant_data, lat='Latitude', lon='Longitude', color=selected_pollutant,
                            hover_name='City or Locality',
                            hover_data=[selected_pollutant, 'PM10 (μg/m3)', 'NO2 (μg/m3)'],
                            title=f'Zanieczyszczenie powietrza w kraju: {selected_country}, rok: {selected_year} ({selected_pollutant})',
                            mapbox_style='carto-positron', zoom=3, height=600,
                            range_color=[country_year_pollutant_data[selected_pollutant].min(),
                                         country_year_pollutant_data[selected_pollutant].max()],
                            center={'lat': country_year_pollutant_data['Latitude'].mean(),
                                    'lon': country_year_pollutant_data['Longitude'].mean()})

    fig.show()


def update_pollutants_options(selected_country):
    available_pollutants = ['PM2.5 (μg/m3)', 'PM10 (μg/m3)', 'NO2 (μg/m3)']
    select_pollutant.options = available_pollutants

    if select_pollutant.value not in available_pollutants:
        select_pollutant.value = available_pollutants[0]

    available_countries = dane[dane[selected_pollutant].notna()]['WHO Country Name'].unique()
    select_country.options = available_countries

    if select_country.value not in available_countries:
        select_country.value = available_countries[0]


def update_years_options(selected_country, selected_pollutant):
    available_years = sorted(dane[(dane['WHO Country Name'] == selected_country) &
                                  (dane[selected_pollutant].notna())]['Measurement Year'].unique())
    select_year.options = available_years

    if select_year.value not in available_years:
        select_year.value = available_years[0]


americas_countries = dane[dane['WHO Region'] == 'Region of the Americas']['WHO Country Name'].unique()

select_pollutant = widgets.Dropdown(description='Czynnik:', style={'description_width': 'initial'})

select_country = widgets.Dropdown(options=americas_countries, description='Kraj:',
                                  style={'description_width': 'initial'})

select_year = widgets.Dropdown(description='Rok:', style={'description_width': 'initial'})

widgets.interactive(update_years_options, selected_country=select_country, selected_pollutant=select_pollutant)
widgets.interactive(update_pollutants_options, selected_country=select_country)

widgets.interactive(update_map, selected_pollutant=select_pollutant, selected_country=select_country,
                    selected_year=select_year)


def update_map(selected_pollutant):
    southeast_asia_data = dane[dane['WHO Region'] == 'South East Asia Region']
    southeast_asia_data_filtered = southeast_asia_data.dropna(subset=[selected_pollutant])
    southeast_asia_data_sorted = southeast_asia_data_filtered.sort_values(by='Measurement Year')

    fig = px.scatter_mapbox(southeast_asia_data_sorted, lat='Latitude', lon='Longitude', color=selected_pollutant,
                            hover_name='City or Locality', hover_data=[selected_pollutant, 'PM10 (μg/m3)', 'NO2 (μg/m3)'],
                            animation_frame='Measurement Year',
                            title=f'Zanieczyszczenie powietrza w regionie Południowo-Wschodniej Azji na przestrzeni lat 2010-2020 ({selected_pollutant})',
                            mapbox_style='carto-positron', zoom=3, height=750,
                            range_color=[southeast_asia_data_sorted[selected_pollutant].min(), southeast_asia_data_sorted[selected_pollutant].max()],
                            center={'lat': southeast_asia_data_sorted['Latitude'].mean(), 'lon': southeast_asia_data_sorted['Longitude'].mean()})
    fig.show()

select_pollutant = widgets.Dropdown(options=['PM2.5 (μg/m3)', 'PM10 (μg/m3)', 'NO2 (μg/m3)'], description='Czynnik:', style={'description_width': 'initial'})

widgets.interactive(update_map, selected_pollutant=select_pollutant)


def update_map(selected_pollutant, selected_country, selected_year):
    country_year_pollutant_data = dane[(dane['WHO Country Name'] == selected_country) &
                                       (dane['Measurement Year'] == selected_year) &
                                       (dane[selected_pollutant].notna())]

    fig = px.scatter_mapbox(country_year_pollutant_data, lat='Latitude', lon='Longitude', color=selected_pollutant,
                            hover_name='City or Locality',
                            hover_data=[selected_pollutant, 'PM10 (μg/m3)', 'NO2 (μg/m3)'],
                            title=f'Zanieczyszczenie powietrza w kraju: {selected_country}, rok: {selected_year} ({selected_pollutant})',
                            mapbox_style='carto-positron', zoom=3, height=600,
                            range_color=[country_year_pollutant_data[selected_pollutant].min(),
                                         country_year_pollutant_data[selected_pollutant].max()],
                            center={'lat': country_year_pollutant_data['Latitude'].mean(),
                                    'lon': country_year_pollutant_data['Longitude'].mean()})

    fig.show()


def update_pollutants_options(selected_country):
    available_pollutants = ['PM2.5 (μg/m3)', 'PM10 (μg/m3)', 'NO2 (μg/m3)']
    select_pollutant.options = available_pollutants

    if select_pollutant.value not in available_pollutants:
        select_pollutant.value = available_pollutants[0]

    available_countries = dane[dane[selected_pollutant].notna()]['WHO Country Name'].unique()
    select_country.options = available_countries

    if select_country.value not in available_countries:
        select_country.value = available_countries[0]


def update_years_options(selected_country, selected_pollutant):
    available_years = sorted(dane[(dane['WHO Country Name'] == selected_country) &
                                  (dane[selected_pollutant].notna())]['Measurement Year'].unique())
    select_year.options = available_years

    if select_year.value not in available_years:
        select_year.value = available_years[0]


asia_countries = dane[dane['WHO Region'] == 'South East Asia Region']['WHO Country Name'].unique()

select_pollutant = widgets.Dropdown(description='Czynnik:', style={'description_width': 'initial'})

select_country = widgets.Dropdown(options=asia_countries, description='Kraj:', style={'description_width': 'initial'})

select_year = widgets.Dropdown(description='Rok:', style={'description_width': 'initial'})

widgets.interactive(update_years_options, selected_country=select_country, selected_pollutant=select_pollutant)
widgets.interactive(update_pollutants_options, selected_country=select_country)

widgets.interactive(update_map, selected_pollutant=select_pollutant, selected_country=select_country,
                    selected_year=select_year)


def update_map(selected_pollutant):
    western_pacific_data = dane[dane['WHO Region'] == 'Western Pacific Region']
    western_pacific_data_filtered = western_pacific_data.dropna(subset=[selected_pollutant])
    western_pacific_data_sorted = western_pacific_data_filtered.sort_values(by='Measurement Year')

    fig = px.scatter_mapbox(western_pacific_data_sorted, lat='Latitude', lon='Longitude', color=selected_pollutant,
                            hover_name='City or Locality', hover_data=[selected_pollutant, 'PM10 (μg/m3)', 'NO2 (μg/m3)'],
                            animation_frame='Measurement Year',
                            title=f'Zanieczyszczenie powietrza w regionie Zachodniego Pacyfiku na przestrzeni lat 2010-2020 ({selected_pollutant})',
                            mapbox_style='carto-positron', zoom=3, height=750,
                            range_color=[western_pacific_data_sorted[selected_pollutant].min(), western_pacific_data_sorted[selected_pollutant].max()],
                            center={'lat': western_pacific_data_sorted['Latitude'].mean(), 'lon': western_pacific_data_sorted['Longitude'].mean()})
    fig.show()

select_pollutant = widgets.Dropdown(options=['PM2.5 (μg/m3)', 'PM10 (μg/m3)', 'NO2 (μg/m3)'], description='Czynnik:', style={'description_width': 'initial'})

widgets.interactive(update_map, selected_pollutant=select_pollutant)


def update_map(selected_pollutant, selected_country, selected_year):
    country_year_pollutant_data = dane[(dane['WHO Country Name'] == selected_country) &
                                       (dane['Measurement Year'] == selected_year) &
                                       (dane[selected_pollutant].notna())]

    fig = px.scatter_mapbox(country_year_pollutant_data, lat='Latitude', lon='Longitude', color=selected_pollutant,
                            hover_name='City or Locality',
                            hover_data=[selected_pollutant, 'PM10 (μg/m3)', 'NO2 (μg/m3)'],
                            title=f'Zanieczyszczenie powietrza w kraju: {selected_country}, rok: {selected_year} ({selected_pollutant})',
                            mapbox_style='carto-positron', zoom=3, height=600,
                            range_color=[country_year_pollutant_data[selected_pollutant].min(),
                                         country_year_pollutant_data[selected_pollutant].max()],
                            center={'lat': country_year_pollutant_data['Latitude'].mean(),
                                    'lon': country_year_pollutant_data['Longitude'].mean()})

    fig.show()


def update_pollutants_options(selected_country):
    available_pollutants = ['PM2.5 (μg/m3)', 'PM10 (μg/m3)', 'NO2 (μg/m3)']
    select_pollutant.options = available_pollutants

    if select_pollutant.value not in available_pollutants:
        select_pollutant.value = available_pollutants[0]

    available_countries = dane[dane[selected_pollutant].notna()]['WHO Country Name'].unique()
    select_country.options = available_countries

    if select_country.value not in available_countries:
        select_country.value = available_countries[0]


def update_years_options(selected_country, selected_pollutant):
    available_years = sorted(dane[(dane['WHO Country Name'] == selected_country) &
                                  (dane[selected_pollutant].notna())]['Measurement Year'].unique())
    select_year.options = available_years

    if select_year.value not in available_years:
        select_year.value = available_years[0]


pacific_countries = dane[dane['WHO Region'] == 'Western Pacific Region']['WHO Country Name'].unique()

select_pollutant = widgets.Dropdown(description='Czynnik:', style={'description_width': 'initial'})

select_country = widgets.Dropdown(options=pacific_countries, description='Kraj:',
                                  style={'description_width': 'initial'})

select_year = widgets.Dropdown(description='Rok:', style={'description_width': 'initial'})

widgets.interactive(update_years_options, selected_country=select_country, selected_pollutant=select_pollutant)
widgets.interactive(update_pollutants_options, selected_country=select_country)

widgets.interactive(update_map, selected_pollutant=select_pollutant, selected_country=select_country,
                    selected_year=select_year)



# Wykres słupkowy średnich stężeń zanieczyszczeń w poszczególnych regionach
mean_pm25_region = dane.groupby('WHO Region')['PM2.5 (μg/m3)'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=mean_pm25_region, x='WHO Region', y='PM2.5 (μg/m3)')
plt.title('Średnie stężenia PM2.5 w poszczególnych regionach - za okres 2010-2021')
plt.xlabel('Region')
plt.xticks(fontsize=7)
plt.ylabel('Średnie PM2.5 (μg/m3)')
plt.show()

mean_pm10_region = dane.groupby('WHO Region')['PM10 (μg/m3)'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=mean_pm10_region, x='WHO Region', y='PM10 (μg/m3)')
plt.title('Średnie stężenia PM10 w poszczególnych regionach - za okres 2010-2021')
plt.xlabel('Region')
plt.xticks(fontsize=7)
plt.ylabel('Średnie PM10 (μg/m3)')
plt.show()

# Policzenie średnich stężeń dla Western Pacific Region dla PM2.5 i PM10
mean_pm25_western_pacific = dane[dane['WHO Region'] == 'Western Pacific Region']['PM2.5 (μg/m3)'].mean()
mean_pm10_western_pacific = dane[dane['WHO Region'] == 'Western Pacific Region']['PM10 (μg/m3)'].mean()
print("Średnie stężenie PM2.5 dla Western Pacific Region:", mean_pm25_western_pacific)
print("Średnie stężenie PM10 dla Western Pacific Region:", mean_pm10_western_pacific)
count_pm25_western_pacific = dane[dane['WHO Region'] == 'Western Pacific Region']['PM2.5 (μg/m3)'].count()
count_pm10_western_pacific = dane[dane['WHO Region'] == 'Western Pacific Region']['PM10 (μg/m3)'].count()
print("Liczba danych dla PM2.5 w Western Pacific Region:", count_pm25_western_pacific)
print("Liczba danych dla PM10 w Western Pacific Region:", count_pm10_western_pacific)

mean_no2_region = dane.groupby('WHO Region')['NO2 (μg/m3)'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=mean_no2_region, x='WHO Region', y='NO2 (μg/m3)')
plt.title('Średnie stężenia NO2 w poszczególnych regionach - za okres 2010-2021')
plt.xlabel('Region')
plt.ylabel('Średnie NO2 (μg/m3)')
plt.xticks(fontsize=7)
plt.show()

# Wykres słupkowy dla średnich stężeń zanieczyszczeń w poszczególnych regionach
plt.figure(figsize=(15, 10))

colors = sns.color_palette('husl', len(dane['Measurement Year'].unique()))

for i, pollutant in enumerate(['PM2.5 (μg/m3)', 'PM10 (μg/m3)', 'NO2 (μg/m3)']):
    # Grupowanie danych po roku i regionie oraz obliczenie średnich wartości dla danego roku i regionu
    mean_pollutant_year_region = dane.groupby(['Measurement Year', 'WHO Region'])[pollutant].mean().reset_index()

    plt.subplot(3, 1, i + 1)
    sns.barplot(data=mean_pollutant_year_region, x='WHO Region', y=pollutant, hue='Measurement Year', palette=colors)
    plt.title(
        f'Średnie stężenia {pollutant} w poszczególnych regionach w latach {dane["Measurement Year"].min()}-{dane["Measurement Year"].max()}')
    plt.xlabel('Region')
    plt.ylabel(f'Średnie {pollutant}')
    plt.legend(title='Rok', loc='upper right')
    plt.tight_layout()

plt.show()

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

plt.figure(figsize=(12, 8))

for i, pollutant in enumerate(['PM2.5 (μg/m3)', 'PM10 (μg/m3)', 'NO2 (μg/m3)']):
    plt.figure(figsize=(10, 6))

    for j, region in enumerate(dane['WHO Region'].unique()):
        # Wybór danych dla danego regionu i zanieczyszczenia
        region_data = dane[dane['WHO Region'] == region]
        pollutant_data = region_data.groupby('Measurement Year')[pollutant].mean()

        # Tworzenie wykresu liniowego
        plt.plot(pollutant_data.index, pollutant_data.values, label=region, color=colors[j], marker='o')

    # Konfiguracja wykresu
    plt.title(
        f'Zmiana średniego stężenia {pollutant} w poszczególnych regionach w latach {dane["Measurement Year"].min()}-{dane["Measurement Year"].max()}')
    plt.xlabel('Rok')
    plt.ylabel(f'Średnie {pollutant}')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def create_bar_chart(selected_pollutant, selected_region):
    # Filtracja danych dla wybranego regionu i czynnika
    selected_data = dane[dane['WHO Region'] == selected_region][['WHO Country Name', selected_pollutant]].dropna()

    # Zsumowanie średnich wartości dla każdego kraju
    summed_data = selected_data.groupby('WHO Country Name').mean().reset_index()

    fig = px.bar(summed_data, x='WHO Country Name', y=selected_pollutant,
                 title=f'Średnie zanieczyszczenie {selected_pollutant} w krajach regionu {selected_region}',
                 labels={selected_pollutant: f'Średnie {selected_pollutant}', 'WHO Country Name': 'Kraj'},
                 color_discrete_sequence=px.colors.qualitative.Plotly)

    fig.show()


# Wybór czynnika i regionu
select_pollutant = widgets.Dropdown(options=['PM2.5 (μg/m3)', 'PM10 (μg/m3)', 'NO2 (μg/m3)'],
                                    description='Wybierz czynnik:')
select_region = widgets.Dropdown(options=dane['WHO Region'].unique(), description='Wybierz region:')

widgets.interactive(create_bar_chart, selected_pollutant=select_pollutant, selected_region=select_region)



selected_pollutant = 'PM2.5 (μg/m3)'

grouped_data = dane.groupby('WHO Country Name')

# Filtrowanie danych, aby wybrać tylko kraje mające co najmniej 30 rekordów dla danego czynnika
filtered_data = grouped_data.filter(lambda x: (x[selected_pollutant].notna().sum()) >= 30)

# Pogrupowanie ponownie po kraju i obliczenie średniej wartości dla wybranego czynnika
average_pollution_by_country = filtered_data.groupby('WHO Country Name')[selected_pollutant].mean().reset_index()

# Sortowanie krajów na podstawie średniej wartości czynnika zanieczyszczenia
sorted_countries = average_pollution_by_country.sort_values(by=selected_pollutant)

top_5_most_polluted_countries = sorted_countries.tail(5)
top_5_least_polluted_countries = sorted_countries.head(5)

print("Najbardziej zanieczyszczone kraje:")
print(top_5_most_polluted_countries)

print("\nNajmniej zanieczyszczone kraje:")
print(top_5_least_polluted_countries)


selected_pollutant = 'PM10 (μg/m3)'

grouped_data = dane.groupby('WHO Country Name')

filtered_data = grouped_data.filter(lambda x: (x[selected_pollutant].notna().sum()) >= 30)

average_pollution_by_country = filtered_data.groupby('WHO Country Name')[selected_pollutant].mean().reset_index()

sorted_countries = average_pollution_by_country.sort_values(by=selected_pollutant)

top_5_most_polluted_countries = sorted_countries.tail(5)
top_5_least_polluted_countries = sorted_countries.head(5)

print("Najbardziej zanieczyszczone kraje:")
print(top_5_most_polluted_countries)

print("\nNajmniej zanieczyszczone kraje:")
print(top_5_least_polluted_countries)


selected_pollutant = 'NO2 (μg/m3)'

grouped_data = dane.groupby('WHO Country Name')

filtered_data = grouped_data.filter(lambda x: (x[selected_pollutant].notna().sum()) >= 30)

average_pollution_by_country = filtered_data.groupby('WHO Country Name')[selected_pollutant].mean().reset_index()

sorted_countries = average_pollution_by_country.sort_values(by=selected_pollutant)

top_5_most_polluted_countries = sorted_countries.tail(5)
top_5_least_polluted_countries = sorted_countries.head(5)

print("Najbardziej zanieczyszczone kraje:")
print(top_5_most_polluted_countries)

print("\nNajmniej zanieczyszczone kraje:")
print(top_5_least_polluted_countries)