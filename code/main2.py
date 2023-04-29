import json

# Opening JSON file
f = open('input/deepfake-detection-challenge/train_sample_videos/metadata.json')

# returns JSON object as
# a dictionary
data = json.load(f)

# Iterating through the json
# list
print(len(data))

# Closing file
f.close()