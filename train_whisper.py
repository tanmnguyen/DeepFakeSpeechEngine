import torch 
import whisper 

# load base model
model = whisper.load_model("base.en")
# save statedict 
torch.save(model.state_dict(), "weights/asr/base_whisper_model.pth")

# load tiny model 
model = whisper.load_model("tiny.en")
# save statedict
torch.save(model.state_dict(), "weights/asr/tiny_whisper_model.pth")