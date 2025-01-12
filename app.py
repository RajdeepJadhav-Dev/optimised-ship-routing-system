import subprocess
import time
import webbrowser

# Paths to your files
python_file = "C:/Users/Ayush Goyal/OneDrive/Documents/latest/input20.py" # Replace with your Python file path
notebook_file = "C:/Users/Ayush Goyal/OneDrive/Documents/shiprouting/ algo.folium.7.tk.ipynb" # Replace with your Jupyter Notebook file path
html_file = "C:/Users/Ayush Goyal/OneDrive/Documents/shiprouting/map.html"      # Replace with your HTML file path

# Run the Python script
print(f"Running {python_file}...")
subprocess.run(["python", python_file], check=True)


# Wait for 1 second
time.sleep(1)

# Run the Jupyter notebook
print(f"Running {notebook_file}...")
#subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", notebook_file], check=True)
#subprocess.run(["python", "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", notebook_file], check=True)
subprocess.run(["python", "-m", "nbconvert", "--to", "notebook", "--execute", notebook_file], check=True)

# Wait for 1 second
time.sleep(1)

# Open the HTML file in the default web browser
print(f"Opening {html_file}...")
webbrowser.open(html_file)