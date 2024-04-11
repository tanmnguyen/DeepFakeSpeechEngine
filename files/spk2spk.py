num_spks = 1000 
spks = []

with open("egs/tedlium/data/train/spk2utt", "r") as file:
    for line in file:
        content = line.split()
        spks.append(content[0])
        if len(spks) == num_spks:
            break

assert len(spks) == num_spks