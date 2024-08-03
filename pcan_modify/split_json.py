import json
import os
f = open('output_MOTS_val_maskdino.json')
data=json.load(f)

f1=open('/projectnb/cs585bp/hsharma/bdd100k_converted/bdd/labels/seg_track_20/seg_track_val_cocoformat.json')
data1=json.load(f1)
videolist=[]
for i in data1['videos']:
    videolist.append(i['name'])

print(videolist)

os.mkdir('result_final_maskdino')
for i in videolist:
    list_final=[]
    frame_dict={}
    path1='result_final_maskdino/'+i+'.json'
    for j in data:
        if(j['videoName']==i):
            list_final.append(j)
    frame_dict["frames"]=list_final
    frame_dict["groups"]=None
    frame_dict["config"]=None
    #"groups": null,
    #"config": null
    json_data = json.dumps(frame_dict)
    with open(path1, "w") as json_file:
        json_file.write(json_data)