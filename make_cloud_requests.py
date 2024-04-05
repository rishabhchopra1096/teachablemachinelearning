import requests

# The URL of the FastAPI endpoint on Render
# Use the link given by Render for your deployed service
endpoint_url = 'https://teachablemachinelearning.onrender.com/classify/'

# The image URL you want to classify
image_url = 'https://oyster.ignimgs.com/mediawiki/apis.ign.com/the-avengers/b/b9/MR.jpg?width=325'

# Append the image URL as a query parameter directly in the endpoint URL
full_url = f'{endpoint_url}?image_url={image_url}'

# Make the GET request to the FastAPI endpoint
response = requests.get(full_url)

# Check if the request was successful
if response.status_code == 200:
    print("Request successful.")
    print("Response:", response.json())
else:
    print("Request failed.")
    print("Status Code:", response.status_code)
    print("Response:", response.text)
