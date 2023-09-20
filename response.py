import requests

url = "https://api.openai.com/v1/models"

payload = {}
headers = {
  'Authorization': 'Bearer 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'
}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)
