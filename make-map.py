import pandas as pd
import numpy as np
import plotly.offline as py
py.init_notebook_mode(connected=True)

#Outputs an HTML file of an interactive map that displays a world map of average temperatures.
def main():
    print("Getting Country Temperature Data...")
    map_data = pd.read_csv("data/GlobalLandTemperaturesByCity.csv")
    map_data.dropna(inplace=True)
    map_data.sort_values(by=['dt'], inplace=True)
    print("Calculating Country Averages...")
    countries = np.unique(map_data['Country'])
    mean_temp = []
    num_countries = len(countries)
    for i,country in enumerate(countries):
        print("Calculating average for "+str(country)+"\t("+str(i+1)+"/"+str(num_countries)+")")
        mean_temp.append(map_data[map_data['Country'] == country]['AverageTemperature'].mean())
    print("Averages calculated! Creating plotly choropleth map...")
        
    data = [ dict(
            type = 'choropleth',
            locations = countries,
            z = mean_temp,
            zmax = 28,
            zmin = -3.2,
            locationmode = 'country names',
            text = countries,
            colorscale = [[0,"#053061"],[0.1,"#2166ac)"],[0.2,"#4393c3"],
            [0.3,"#92c5de"],[0.4,"#d1e5f0)"],[0.5,"#fddbc7"],[0.6,"#f4a582"],[0.7, "#d6604d"],[0.8,"#b2182b"], [0.9,"#67001f"]],
            hoverinfo = "location+z",
            marker = dict(
                line = dict(color = 'rgb(255,255,255)', width = 1)),
                colorbar = dict(autotick = True, tickprefix = '', 
                title = 'Average\nTemperature,\n°C')
                )
           ]
    layout = dict(
        title = 'Overall Average Surface Temperature by Country',
        geo = dict(
            coastlinecolor = "rgb(255,255,255)",
            showframe = False,
            showocean = False,
            projection = dict(
                    type = 'Equirectangular',
            ),
            lonaxis =  dict(showgrid = False),
            lataxis = dict(showgrid = False)
        ),
        hoverlabel = dict(
            bordercolor = 'rgb(255,255,255)'
                ),
    )
    
    fig = dict(data=data, layout=layout)
    py.plot(fig, validate=False, filename='worldmap.html')
    print("worldmap.html created!")
    
main()
