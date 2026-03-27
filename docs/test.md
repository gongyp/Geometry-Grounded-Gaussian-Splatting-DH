# step1: 全量训练
```bash
python train.py -s eval_tnt/GT_TNT_dataset/Meetingroom-Localupdate \
                -m output/Meetingroom-Localupdate \
                -r 2 \
                --use_decoupled_appearance 3
```
# step2：增量更新
```bash
python run_incremental.py --model_path output/Meetingroom-Localupdate/point_cloud/iteration_30000 \
                --source_path eval_tnt/GT_TNT_dataset/Meetingroom-Localupdate \
                --output_dir output/Meetingroom-Localupdate/update \
                --densify True
                --eval  \
                --save_checkpoints
```