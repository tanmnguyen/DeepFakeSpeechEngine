python decode_generator.py --data egs/tedlium/data/ --set train --start_spk_idx 0 --spk weights/spk/spk_model_A.pt
python decode_generator.py --data egs/tedlium/data/ --set train --start_spk_idx 150 --spk weights/spk/spk_model_B.pt
python decode_generator.py --data egs/tedlium/data/ --set train --start_spk_idx 732 --spk weights/spk/spk_model_C.pt
python decode_generator.py --data egs/tedlium/data/ --set train --start_spk_idx 800 --spk weights/spk/spk_model_D.pt