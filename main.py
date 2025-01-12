import streamlit as st
import folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time

st.title("Ship Routing Application")
xx
# Input fields for user data
start_location = st.text_input("Starting Location (City, Country)")
end_location = st.text_input("Ending Location (City, Country)")
ship_speed = st.number_input("Ship Speed (km/h)", min_value=1.0, step=0.1)
fuel_consumption = st.number_input("Fuel Consumption Rate (liters/km)", min_value=0.1, step=0.1)

# Define the ship routing algorithm (replace with your own implementation)
def route_ship(start_coords, end_coords, speed):
    # Implement your ship routing algorithm here
    # This is a placeholder, you need to replace it with your actual algorithm
    # Example:
    return [(start_coords[0], start_coords[1]), (end_coords[0], end_coords[1])]

# Geocoding function to get coordinates from location names
geolocator = Nominatim(user_agent="ship_routing_app")
def get_coordinates(location):
    location = geolocator.geocode(location)
    return (location.latitude, location.longitude)

# Process user inputs and generate the route
if st.button("Calculate Route"):
    # Get coordinates from location names
    start_coords = get_coordinates(start_location)
    end_coords = get_coordinates(end_location)

    # Calculate the optimal route
    route = route_ship(start_coords, end_coords, ship_speed)

    # Calculate total distance, fuel consumption, and travel time
    total_distance = geodesic(start_coords, end_coords).km
    total_fuel_consumption = total_distance * fuel_consumption
    travel_time = total_distance / ship_speed

    # Create a Folium map
    my_map = folium.Map(location=start_coords, zoom_start=5)

    # Add markers for start and end locations
    folium.Marker(start_coords, popup=start_location).add_to(my_map)
    folium.Marker(end_coords, popup=end_location).add_to(my_map)

    # Add the ship's route as a polyline
    folium.PolyLine(route, color="blue", weight=2.5, opacity=1).add_to(my_map)

    # Display the map in Streamlit
    st.write(my_map)

    # Display the calculated values
    st.write(f"Total Distance: {total_distance:.2f} km")
    st.write(f"Fuel Consumption: {total_fuel_consumption:.2f} liters")
    st.write(f"Travel Time: {travel_time:.2f} hours")