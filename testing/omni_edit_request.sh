#!/bin/bash

# Base64 encode the image file and create JSON directly for curl
IMAGE_B64=$(base64 -w 0 "./omnigen.png")

# Create temporary JSON file
TEMP_JSON=$(mktemp)

cat > "$TEMP_JSON" << EOM
{
  "id" : "1",
    "parameters": {
    "binary_data_output": false
  },
  "inputs" : [
    {
      "name":"6_text",
      "datatype": "BYTES",
      "shape": [1],
      "data": ["white walkers, coding with a laprop, veryfrostrated"],
      "parameters":{
        "name":"prompt"
      }
    },
    {
      "name":"16_image",
      "datatype": "BYTES",
      "shape": [1],
      "data": ["$IMAGE_B64"],
      "parameters":{
        "as_base64": true
      }
    }
  ],
  "outputs" : [
    {
      "name": "8_0_IMAGE",
      "datatype": "FP32",
      "shape": [-1],
      "parameters": {
        "rename":"image",
        "binary_data": false,
        "to_base64":true
      }
    }
  ]
}
EOM

# Make the API call using the temp file
curl -H "Content-Type: application/json" \
     localhost:8080/v2/models/image_omnigen2_image_edit/infer \
     -d @"$TEMP_JSON" | jq -r '.outputs[0].data[0]' | base64 -d > generated_image_curl.png

# Clean up
rm "$TEMP_JSON"
