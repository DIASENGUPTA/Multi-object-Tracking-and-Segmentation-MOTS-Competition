import json
with open('seg_track_new_test_cocoformat.json') as f:
    data = json.load(f)

for item in data['annotations']:
    if(item['category_id']==2):
        item['category_id'] = 1

with open('modifiedjson.json', 'w') as f:
    json.dump(data, f)