#!/bin/bash
#"CowDataset" "GiraffeZebraID" "Giraffes" "HyenaID2022" "IPanda50" "LeopardID2022" "NyalaData" "PolarBearVidID" "SealID" "StripeSpotter" "SeaStarReID2023" "SeaTurtleID" "WhaleSharkID" "BelugaID" "NDD20"
#params=("atrw" "cowdataset" "elpephants" "ctai" "chicks4freeid" "sealid" "seastarreid2023")


#python main.py --ds atrw --train --method ensamble --save_eval --use_geometric_verification --use_lightglue --use_global_embedding --embedding_model megadescriptor-l-384 --remove_background --use_mantiuk --use_fisher --use_md_baseline_split --debug --fusion_signals global fisher gv
#python main.py --ds cowdataset elpephants --train --method ensamble --save_eval --use_geometric_verification --use_lightglue --use_global_embedding --embedding_model megadescriptor-l-384 --remove_background --use_mantiuk --use_fisher --debug --fusion_signals global fisher gv
#python main.py --ds chicks4freeid --train --method ensamble --save_eval --use_geometric_verification --use_lightglue --use_global_embedding --use_fisher --debug --fusion_signals global fisher gv
python main.py --ds ctai --use_mantiuk --train --method ensamble --save_eval --use_geometric_verification --use_lightglue --use_global_embedding --use_fisher --use_md_baseline_split --debug --fusion_signals global fisher gv
python main.py --ds seastarreid2023 --train --method ensamble --save_eval --use_geometric_verification --use_lightglue --use_global_embedding --use_fisher --remove_background --debug --fusion_signals global fisher gv
#sealid
python main.py --ds atrw --train --method ensamble --save_eval --use_geometric_verification --use_lightglue --use_global_embedding --embedding_model megadescriptor-l-384 --remove_background --use_mantiuk --use_fisher --use_md_baseline_split --debug --fusion_signals global fisher gv
python main.py --ds cowdataset elpephants --train --method ensamble --save_eval --use_geometric_verification --use_lightglue --use_global_embedding --embedding_model megadescriptor-l-384 --remove_background --use_mantiuk --use_fisher --debug --fusion_signals global fisher gv
python main.py --ds chicks4freeid --train --method ensamble --save_eval --use_geometric_verification --use_lightglue --use_global_embedding --use_fisher --debug --fusion_signals global fisher gv
python main.py --ds ctai sealid --train --method ensamble --save_eval --use_geometric_verification --use_lightglue --use_global_embedding --use_fisher --use_md_baseline_split --debug --fusion_signals global fisher gv
python main.py --ds seastarreid2023 --train --method ensamble --save_eval --use_geometric_verification --use_lightglue --use_global_embedding --use_fisher --remove_background --debug --fusion_signals global fisher gv
