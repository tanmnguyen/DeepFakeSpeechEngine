spks = set() 
filepath = "egs/tedlium/data/train/spk2utt"

with open(filepath, "r") as file:
    for line in file:
        content = line.split()
        spk_id = content[0]
        spk_id = spk_id.split("-")[0].split("_")[0]
        spks.add(spk_id)

print(spks)
print(len(spks))
