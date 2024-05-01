python decode_generator.py --data egs/tedlium/data/ --set train --start_spk_idx 0 --weight weights/gen/gen_model_A.pt
python decode_generator.py --data egs/tedlium/data/ --set train --start_spk_idx 150 --weight weights/gen/gen_model_B.pt
python decode_generator.py --data egs/tedlium/data/ --set train --start_spk_idx 732 --weight weights/gen/gen_model_C.pt
python decode_generator.py --data egs/tedlium/data/ --set train --start_spk_idx 800 --weight weights/gen/gen_model_D.pt