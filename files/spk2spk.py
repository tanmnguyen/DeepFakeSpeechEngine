# num_spks = 1e9 # all 
num_spks = 500
spks = set()

with open("egs/tedlium/data/train/spk2utt", "r") as file:
    for line in file:
        content = line.split()
        spk_id = content[0].split("-")[0].split("_")[0]
        spks.add(spk_id)
        if len(spks) == num_spks:
            break

assert len(spks) <= num_spks

spks = list(spks)