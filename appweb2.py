import tkinter as tk
from tkinter import messagebox
from geopy.distance import geodesic
import folium
import webbrowser
import os
import xarray as xr

# Load wave data from the dataset
from skimage.graph import route_through_array
import xarray as xr
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot 
import numpy as np
import folium 
import json
from geopy.geocoders import Nominatim
# Initialize the Nominatim geocoder with a descriptive user-agent
# geolocator = Nominatim(user_agent="my_app_name/1.0 (my_email@example.com)")
from geopy.distance import geodesic







# %%
# Get array index to the value that is closest to a given value
def get_closest(array, value):
    return np.abs(array - value).argmin()

# %% [markdown]
# ## Load CMEMS wave analysis/forecast data

# %%
file_path = "C:/Users/Ayush Goyal/OneDrive/Documents/shiprouting/cmems_mod_glo_wav_anfc_0.083deg_PT3H-i_1724952435266.nc"

dataset_waves= xr.open_dataset(file_path)
dataset_waves

# %%
file_path = "C:/Users/Ayush Goyal/OneDrive/Documents/shiprouting/cmems_mod_glo_phy_anfc_0.083deg_P1M-m_1724955162581.nc"

dataset_physics= xr.open_dataset(file_path)
print(dataset_physics)



# %% [markdown]
# 

# %%
# microwave sea ice

file_path =  "C:/Users/Ayush Goyal/OneDrive/Documents/shiprouting/microwaveasmr2seaice.nc"


# Open the dataset without decoding the time variable
dataset_ice = xr.open_dataset(file_path, decode_times=False)
print(dataset_ice)


# %%
# microwave temp

file_path =  "C:/Users/Ayush Goyal/OneDrive/Documents/shiprouting/microwaveasmr2temp.nc"


# Open the dataset without decoding the time variable
dataset_microwavetemp = xr.open_dataset(file_path, decode_times=False)
print(dataset_microwavetemp)


# %%
# microwave rain

file_path =  "C:/Users/Ayush Goyal/OneDrive/Documents/shiprouting/microwaveasmr2rain.nc"


# Open the dataset without decoding the time variable
dataset_microwaverain = xr.open_dataset(file_path, decode_times=False)
print(dataset_microwaverain)

microrain = dataset_microwaverain.rain_rate.isel(
    time= 0 ,       )    # Specify the appropriate time slice


microrain.data[np.isnan(microrain.data)] = 1

print(microrain.data)




# %%
# microwave land mask

file_path =  "C:/Users/Ayush Goyal/OneDrive/Documents/shiprouting/microwavelandmask.nc"


# Open the dataset without decoding the time variable
dataset_microwavelandmask = xr.open_dataset(file_path, decode_times=False)
print(dataset_microwavelandmask )


# %% [markdown]
# 
# ## Define routing problem
# Calculate the optimal shipping route between New York and Lisbon avoiding high waves.
# 
# New York: 40.7128° N, 74.0060° W
# Lisbon: 38.7223° N, 9.1393° W
# 

# %% [markdown]
# ## Plotting graphs

# %%
# # Plot Wave Height variable in the dataset for a specific time slice
# dataset_waves.VHM0.isel(time=0).plot(robust=True, aspect=2, size=7);

# %%
# # Plot Mean Wave Direction variable in the dataset for a specific time slice
# dataset_waves.VMDR.isel(time=0).plot(robust=True, aspect=2, size=7);

# %% [markdown]
# ## Define area of interest

# %%
# Set bounding box for the allowed routing corridor
bbox = ((-90, -180), (90, 180))
# Select time
time_slice = 0

# %%
# Get indices of the bbox
lon_min = get_closest(dataset_microwaverain.lon.data, bbox[0][0])
lat_min = get_closest(dataset_microwaverain.lat.data, bbox[0][1])
lon_max = get_closest(dataset_microwaverain.lon.data, bbox[1][0])
lat_max = get_closest(dataset_microwaverain.lat.data, bbox[1][1])

# %% [markdown]
# ## Extracting and slicing DataSets 

# %%
# wave height copernicus marine

# Extract array  of wave height from dataset_waves to define the cost in the routing algorithm 
# -> subset space, time and variable
wave_height = dataset_waves.VHM0.isel(time=time_slice, longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))

wave_height.data[np.isnan(wave_height.data)] = 99
# wave_height.plot(robust=True, aspect=2, size=7);

# %%
#wind wave height copernicus marine

# Extract array of wind wave height from dataset_waves to define the cost in the routing algorithm 
# -> subset space, time and variable
wind_wave_height = dataset_waves.VHM0_WW.isel(time=time_slice, longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))

wind_wave_height.data[np.isnan(wind_wave_height.data)] = 99
# wind_wave_height.plot(robust=True, aspect=2, size=7);

# %% [markdown]
# 

# %%

sea_mixed_layer_thick=dataset_physics.mlotst.isel(
    time=time_slice,
    longitude=slice(lon_min, lon_max),
    latitude=slice(lat_min, lat_max))


# %%
sea_water_salinity = dataset_physics.sob.isel(time=time_slice, longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))
# sea_water_salinity.plot(robust=True, aspect=2, size=7);

# %%
sea_water_temperature = dataset_physics.tob.isel(time=time_slice, longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))
# sea_water_temperature.plot(robust=True, aspect=2, size=7);
# np.shape(sea_water_temperature.data)

# %%
eastward_sea_ice_velocity = dataset_physics.usi.isel(time=time_slice, longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))
# sea_water_temperature.plot(robust=True, aspect=2, size=7);
# np.shape(sea_water_temperature.data)

# %%
northward_sea_ice_velocity = dataset_physics.vsi.isel(time=time_slice, longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))
# sea_water_temperature.plot(robust=True, aspect=2, size=7);
# np.shape(sea_water_temperature.data)

# %%
northward_sea_ice_velocity = dataset_physics.vsi.isel(time=time_slice, longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))
# sea_water_temperature.plot(robust=True, aspect=2, size=7);
# np.shape(sea_water_temperature.data)

# %%
sea_surface_height_above_geoid = dataset_physics.zos.isel(time=time_slice, longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))
# sea_water_temperature.plot(robust=True, aspect=2, size=7);
# np.shape(sea_water_temperature.data)
print(sea_surface_height_above_geoid.data)


# %%
ice = dataset_ice.sea_ice_mask.isel(
    time=time_slice,           # Specify the appropriate time slice
    lat=slice(lat_min, lat_max),  # Latitude range
    lon=slice(lon_min, lon_max))   # Longitude range

ice.data[np.isnan(ice.data)] = 99


# %%
microtemp = dataset_microwavetemp.SST.isel(
    time=time_slice,           # Specify the appropriate time slice
    lat=slice(lat_min, lat_max),  # Latitude range
    lon=slice(lon_min, lon_max))   # Longitude range

microtemp.data[np.isnan(microtemp.data)] = 99



# %%
microrain = dataset_microwaverain.rain_rate.isel(
    time=time_slice,           # Specify the appropriate time slice
    lat=slice(lat_min, lat_max),  # Latitude range
    lon=slice(lon_min, lon_max))   # Longitude range

microrain.data[np.isnan(microrain.data)] = 99
print(microrain.data)



# %%
microland = dataset_microwavelandmask.land_mask.isel(
    time=time_slice,           # Specify the appropriate time slice
    lat=slice(lat_min, lat_max),  # Latitude range
    lon=slice(lon_min, lon_max))   # Longitude range
microland.data = microland.data.astype(np.int16)

microland.data[microland.data > 0] = 9999
print(microland.data)



# %%
# copernicus physics
# Extract array of mixed layer thickness from dataset_waves to define the cost in the routing algorithm 
# -> subset space, time and variable
# sea_mixed_layer_thick = dataset_physics.mlotst.isel(time=time_slice, longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))
# sea_mixed_layer_thick.plot(robust=True, aspect=2, size=7);

# %%
np.shape(ice.data)

# %%
np.shape(microrain.data)

# %%
np.shape(microtemp)

# %%
np.shape(wind_wave_height.data)

# %%
np.shape(wave_height.data)

# %%


# %% [markdown]
# ## COSTS

# %%
costs = ice.data + microrain.data + microland.data + wind_wave_height.data + wave_height.data + + sea_mixed_layer_thick.data + sea_water_salinity.data + sea_water_temperature.data + northward_sea_ice_velocity.data + eastward_sea_ice_velocity.data + sea_surface_height_above_geoid.data
print(costs)

# %%

# Set NaN values to large costs as the algorithm cannot handle NaNs
costs = costs.astype(np.int16)
costs[np.isnan(costs)] = 999
np.shape(costs)

# %%
print(costs)

# %%


# %%
np.shape(wave_height)

# %% [markdown]
# 
# ## Define start/end point of route

# %%
# # Assuming 'lon' and 'lat' are the correct coordinate names in your dataset
# start_lon = get_closest(wave_height.longitude.data, lon_NY)
# start_lat = get_closest(wave_height.latitude.data, lat_NY)
# end_lon = get_closest(wave_height.longitude.data, lon_LS)
# end_lat = get_closest(wave_height.latitude.data, lat_LS)

# %%
lat_NY =-34.603722
lon_NY = -58.381592
lat_LS = 18.9633
lon_LS = 72.8358



# Assuming 'lon' and 'lat' are the correct coordinate names in your dataset
start_lon = get_closest(microrain.lon.data, lon_NY)
start_lat = get_closest(microrain.lat.data, lat_NY)
end_lon = get_closest(microrain.lon.data, lon_LS)
end_lat = get_closest(microrain.lat.data, lat_LS)

start = (start_lat, start_lon)
end = (end_lat, end_lon)

print(start)

# %% [markdown]
# ## Calculate optimal route (minimum cost path)
# Calculate optimal route based on the minimum cost path
# 

# %%
# Optional parameters:
# - fully_connected 
#     - False -> only axial moves are allowed
#     - True  -> diagonal moves are allowed
# - geometric 
#     - False -> minimum cost path
#     - True  -> distance-weighted minimum cost path

indices, weight = route_through_array(costs, start, end, fully_connected=True, geometric=True)
indices = np.stack(indices, axis=-1)




# %%
# Plot optimal route

# plt.figure(figsize=(14,7))
# # Costs
# plt.imshow(costs, aspect='auto', vmin=np.min(costs), vmax=0.5*np.max(costs));
# # Route
# plt.plot(indices[1],indices[0], 'r')
# # Start/end points
# plt.plot(start_lon, start_lat, 'k^', markersize=15)
# plt.plot(end_lon, end_lat, 'k*', markersize=15)
# plt.gca().invert_yaxis();

# %%
# Indices provided
# indices = np.array([[  0,   1,   2, ..., 104, 105, 105],
#                     [899, 899, 899, ...,  73,  73,  72]])

# Extract latitude and longitude arrays from the dataset
latitudes = dataset_waves.latitude.values
longitudes = dataset_waves.longitude.values
# indices = np.array([[0, 1, 2, 3, 4, 5, 6, 104, 105, 105],  # example values
#                     [899, 899, 899, 899, 899, 899, 899, 73, 73, 72]])  # example values
# indices = indices.astype(int)


# Convert indices to coordinates
latitude_coords = latitudes[indices[0, :]]
longitude_coords = longitudes[indices[1, :]]

# Combine latitude and longitude into coordinate pairs
coordinates = np.vstack((latitude_coords, longitude_coords)).T

# Print the coordinates
for lat, lon in coordinates:
    print(f"Latitude: {lat}, Longitude: {lon}")

# %%


# Extract the indices from the array (assuming 'indices' is the variable holding them)
latitude_indices = indices[0]  # First row corresponds to latitude indices
longitude_indices = indices[1]  # Second row corresponds to longitude indices

# Retrieve the latitude and longitude data from the wave_height dataset
latitudes = microrain.lat.data[latitude_indices]
longitudes = microrain.lon.data[longitude_indices]

# Initialize the map centered around the midpoint of the route
map_center = [np.mean(latitudes), np.mean(longitudes)]
route_map = folium.Map(location=map_center, zoom_start=5)

# Combine latitudes and longitudes into pairs of coordinates
coordinates = list(zip(latitudes, longitudes))


# Ship routing algorithm using wave data
def calculate_optimal_route(wave_data, start_lat, start_lon, end_lat, end_lon):
    """
    This function calculates the optimal route based on the provided wave data.
    Replace the content with your algorithm from the Jupyter notebook.
    """
    # Placeholder: Add your algorithm for calculating the optimal route here
    # For simplicity, we're using a straight line (geodesic) for now
    route = [(start_lat, start_lon), (end_lat, end_lon)]  # Placeholder route
    return route














# Function to calculate route details
def calculate_route():
    
        # Get user inputs from the Tkinter window
        start_lat = float(entry_start_lat.get())
        start_lon = float(entry_start_lon.get())
        end_lat = float(entry_end_lat.get())
        end_lon = float(entry_end_lon.get())
        ship_speed = float(entry_ship_speed.get())
        fuel_rate = float(entry_fuel_rate.get())

        # Load wave data


        start = (start_lat, start_lon)
end = (end_lat, end_lon)

print(start)

# %% [markdown]
# ## Calculate optimal route (minimum cost path)
# Calculate optimal route based on the minimum cost path
# 

# %%
# Optional parameters:
# - fully_connected 
#     - False -> only axial moves are allowed
#     - True  -> diagonal moves are allowed
# - geometric 
#     - False -> minimum cost path
#     - True  -> distance-weighted minimum cost path

indices, weight = route_through_array(costs, start, end, fully_connected=True, geometric=True)
indices = np.stack(indices, axis=-1)




# %%
# Plot optimal route

# plt.figure(figsize=(14,7))
# # Costs
# plt.imshow(costs, aspect='auto', vmin=np.min(costs), vmax=0.5*np.max(costs));
# # Route
# plt.plot(indices[1],indices[0], 'r')
# # Start/end points
# plt.plot(start_lon, start_lat, 'k^', markersize=15)
# plt.plot(end_lon, end_lat, 'k*', markersize=15)
# plt.gca().invert_yaxis();

# %%
# Indices provided
# indices = np.array([[  0,   1,   2, ..., 104, 105, 105],
#                     [899, 899, 899, ...,  73,  73,  72]])

# Extract latitude and longitude arrays from the dataset
latitudes = dataset_waves.latitude.values
longitudes = dataset_waves.longitude.values
# indices = np.array([[0, 1, 2, 3, 4, 5, 6, 104, 105, 105],  # example values
#                     [899, 899, 899, 899, 899, 899, 899, 73, 73, 72]])  # example values
# indices = indices.astype(int)


# Convert indices to coordinates
latitude_coords = latitudes[indices[0, :]]
longitude_coords = longitudes[indices[1, :]]

# Combine latitude and longitude into coordinate pairs
coordinates = np.vstack((latitude_coords, longitude_coords)).T

# Print the coordinates
for lat, lon in coordinates:
    print(f"Latitude: {lat}, Longitude: {lon}")

# %%


# Extract the indices from the array (assuming 'indices' is the variable holding them)
latitude_indices = indices[0]  # First row corresponds to latitude indices
longitude_indices = indices[1]  # Second row corresponds to longitude indices

# Retrieve the latitude and longitude data from the wave_height dataset
latitudes = microrain.lat.data[latitude_indices]
longitudes = microrain.lon.data[longitude_indices]

# Initialize the map centered around the midpoint of the route
map_center = [np.mean(latitudes), np.mean(longitudes)]
route_map = folium.Map(location=map_center, zoom_start=5)

# Combine latitudes and longitudes into pairs of coordinates
coordinates = list(zip(latitudes, longitudes))



























        # Calculate total distance (using straight line for now)
# Calculate the total distance of the route
total_distance = 0
for i in range(1, len(coordinates)):
    total_distance += geodesic(coordinates[i-1], coordinates[i]).kilometers

print(f"Total distance: {total_distance:.2f} km")


# %%
# Function to calculate fuel consumption
def calculate_fuel_consumption(total_distance, fuel_consumption_rate):
    fuel_consumption = total_distance * fuel_consumption_rate
    return fuel_consumption

# Example: User inputs

fuel_consumption_rate = float(input("Enter the fuel consumption rate (liters/km or gallons/mile): "))

# Calculate total fuel consumption
total_fuel = calculate_fuel_consumption(total_distance, fuel_consumption_rate)

# Display result
print(f"Total fuel consumption for the journey: {total_fuel:.2f} units")


# %%
# User input for ship speed (in km/h or another unit as appropriate)
ship_speed = float(input("Enter the ship's speed (in km/h): "))

def calculate_travel_time(total_distance, speed):
    # Time = Distance / Speed
    return total_distance / speed

# Calculate the travel time
travel_time_hours = calculate_travel_time(total_distance, ship_speed)

# Display result
print(f"Total travel time for the journey: {travel_time_hours:.2f} hours")
















# Create a Folium map centered on the starting point
start_location = [latitudes[0], longitudes[0]]  # Starting point of the route
map_route = folium.Map(location=start_location, zoom_start=5)

# Add the route to the map
tooltip_text = f"Total Distance: {total_distance:.2f} kms"
folium.PolyLine(locations=coordinates, color="red", weight=5, opacity=1, line_cap="square", line_join="round" , tooltip=tooltip_text,

).add_to(map_route)

# Example: Adding markers with city names and coordinates to the Folium map
start_city_name = "Start City"  # Replace with the actual start city name
end_city_name = "End City"      # Replace with the actual end city name

# Add a marker for the starting point with the city name and coordinates
folium.Marker(
    location=[latitudes[0], longitudes[0]], 
    popup=f"City: {start_city_name}, Coordinates: ({latitudes[0]}, {longitudes[0]})",
    icon=folium.Icon(color="green")
).add_to(map_route)

# Add a marker for the ending point with the city name and coordinates
folium.Marker(
    location=[latitudes[-1], longitudes[-1]], 
    popup=f"City: {end_city_name}, Coordinates: ({latitudes[-1]}, {longitudes[-1]})",
    icon=folium.Icon(color="red")
).add_to(map_route)


from folium.plugins import MiniMap

# Add MiniMap to the map
minimap = MiniMap(toggle_display=True)
map_route.add_child(minimap)

from folium.plugins import Fullscreen

# Add Fullscreen control to the map
fullscreen = Fullscreen()
map_route.add_child(fullscreen)

# Add header text to the map using HTML
header_html = f"""
             <div style="position: fixed; 
                         top: 10px; left: 50px; width: 300px; height: 200px; 
                         background-color: yellolw; z-index:9999; font-size:20px;">
             <p><b>Total Distance: {total_distance:.2f} km</b></p>
             <p><b>Total Fuel Consumption: {total_fuel:.2f} units</b></p>
             <p><b>Time To Reach Destination: {travel_time_hours:.2f} hours</b></p>
             </div>
             """

# Add the header to the map
map_route.get_root().html.add_child(folium.Element(header_html))

# Display the map
map_route



















        # Save map as HTML
        map_file = "route_map.html"
        map_route.save(map_file)

        # Display results
        result_label.config(text=f"Distance: {total_distance:.2f} km\n"
                                 f"Fuel Consumption: {total_fuel:.2f} liters\n"
                                 f"Travel Time: {total_time:.2f} hours")

        # Open map in browser
        webbrowser.open(map_file)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Tkinter window setup
root = tk.Tk()
root.title("Ship Route Planner")
root.geometry("400x400")

# Labels and entries for user input
tk.Label(root, text="Start Latitude:").pack(pady=5)
entry_start_lat = tk.Entry(root)
entry_start_lat.pack()

tk.Label(root, text="Start Longitude:").pack(pady=5)
entry_start_lon = tk.Entry(root)
entry_start_lon.pack()

tk.Label(root, text="End Latitude:").pack(pady=5)
entry_end_lat = tk.Entry(root)
entry_end_lat.pack()

tk.Label(root, text="End Longitude:").pack(pady=5)
entry_end_lon = tk.Entry(root)
entry_end_lon.pack()

tk.Label(root, text="Ship Speed (km/h):").pack(pady=5)
entry_ship_speed = tk.Entry(root)
entry_ship_speed.pack()

tk.Label(root, text="Fuel Consumption Rate (liters/km):").pack(pady=5)
entry_fuel_rate = tk.Entry(root)
entry_fuel_rate.pack()

# Button to calculate the route
tk.Button(root, text="Calculate Route", command=calculate_route).pack(pady=20)

# Label to display results
result_label = tk.Label(root, text="")
result_label.pack(pady=10)

# Run the Tkinter loop
root.mainloop()
