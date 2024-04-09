import sys 
sys.path.append('../')

import configs 

def compute_error_rate(output, labels):
    # get tokenizer
    tokenizer = configs.speech_recognition_cfg['tokenizer']

    # get normalizer 
    normalizer = configs.speech_recognition_cfg['normalizer']

    # greedy decoding
    output = output.argmax(dim=-1)

    wer, ser, tot_words = 0.0, 0.0, 0.0
    for i in range(output.shape[0]): # batch dim 
        cnt_incorrects = 0
        for j in range(labels.shape[1]): # seq len dim 
            # finish processing if EOS token is found
            if labels[i,j] == tokenizer.eot:
                break
            
            tot_words += 1
            pred_word = normalizer(tokenizer.decode([output[i, j].item()]))
            true_word = normalizer(tokenizer.decode([labels[i, j].item()]))
            if pred_word != true_word:
                cnt_incorrects += 1

        # update WER and SER
        wer += cnt_incorrects
        ser += 1 if cnt_incorrects > 0 else 0

        # debug 
        # if cnt_incorrects > 0:
        #     pred_sent = normalizer(tokenizer.decode(output[i]))
        #     true_sent = normalizer(tokenizer.decode(labels[i]))
        #     print(f"True: {true_sent}")
        #     print(f"Pred: {pred_sent}")

    # normalize WER and SER
    wer /= tot_words
    ser /= output.shape[0]

    return wer, ser



# print(features.shape, tokens.shape, labels.shape)
# print(labels[0])
# tmp = labels[0] 
# tmp[tmp == -100] = tokenizer.eot
# print(tmp)
# print("labels", tokenizer.decode(tmp))

# results = model.decode(features[0].unsqueeze(0))
# print(results)

# exit(0)
# results = model(features, tokens)
# results = torch.argmax(results, dim=-1)

# for res in results:
#     res[res == -100] = tokenizer.eot
#     print(res)
#     text = tokenizer.decode(res)
#     print(text)
#     break
# results = results.argmax(-1)
# print("results", results.shape)
# tmp = results[0]
# tmp[tmp == -100] = tokenizer.eot
# print("output", tokenizer.decode(tmp))