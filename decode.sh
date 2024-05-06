python decode_generator.py --data egs/tedlium/data/ --set train --start_spk_idx 0 --max_len 10 --weight weights/gen/gen_model_A.pt
python decode_generator.py --data egs/tedlium/data/ --set train --start_spk_idx 150 --max_len 10 --weight weights/gen/gen_model_B.pt
python decode_generator.py --data egs/tedlium/data/ --set train --start_spk_idx 732 --max_len 10 --weight weights/gen/gen_model_C.pt
python decode_generator.py --data egs/tedlium/data/ --set train --start_spk_idx 800 --max_len 10 --weight weights/gen/gen_model_D.pt