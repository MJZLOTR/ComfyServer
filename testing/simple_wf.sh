#!/bin/bash

# Base64 encode the image file and create JSON directly for curl

# Create temporary JSON file
TEMP_JSON=$(mktemp)
TEMP_IMG=$(mktemp)


cat > "$TEMP_JSON" << EOM
{
  "id" : "1",
  "inputs" : [
        {"name":"3_steps","datatype":"INT32","shape":[1],"data":[10]},
        {"name":"6_text","datatype":"BYTES","shape":[1],"data":["montains over the city"]}
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
     http://localhost:8080/v2/models/simple-workflow/infer \
     -d @"$TEMP_JSON" | jq -r '.outputs[0].data[0]' > "$TEMP_IMG"
     
cat "$TEMP_IMG" | base64 -d > generated_image_curl.png

# Clean up
rm "$TEMP_JSON" "$TEMP_IMG"
