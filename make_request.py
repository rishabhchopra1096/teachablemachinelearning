import requests

# The URL of the FastAPI endpoint
# Note: You'll include the image URL as a query parameter directly in the endpoint URL
image_url = 'https://oyster.ignimgs.com/mediawiki/apis.ign.com/the-avengers/b/b9/MR.jpg?width=325'
endpoint_url = f'http://127.0.0.1:8000/classify/?image_url={image_url}'

# Make the GET request to the FastAPI endpoint
response = requests.get(endpoint_url)

# Check if the request was successful
if response.status_code == 200:
    print("Request successful.")
    print("Response:", response.json())
else:
    print("Request failed.")
    print("Status Code:", response.status_code)
    print("Response:", response.text)
