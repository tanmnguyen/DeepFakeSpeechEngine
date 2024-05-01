# UNI: tmn2134
import json 

filepath = "egs/tedlium/data/dev/spk2utt"

spks = dict() 
with open(filepath, "r") as file:
    for line in file:
        content = line.split()
        spk_id = content[0]
        spk_id = spk_id.split("-")[0].split("_")[0]
        if spk_id not in spks:
            spks[spk_id] = len(spks)


# dump to json file
json.dump(spks, open("spk2idx.json", "w"), indent=4)