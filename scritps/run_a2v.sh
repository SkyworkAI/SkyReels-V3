export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_PATH="/usr/models/audio2video/a2v_opensource"
PROMPT="A young girl's eyes kept following the camera, singing with a mysterious expression. Surrounded by rich blooming florals, the camera cranes up slowly to reveal the full surreal, luxurious scene."
INPUT_IMAGE="https://skyreels-api.oss-accelerate.aliyuncs.com/examples/talking_avatar_video/single1.png"
INPUT_AUDIO="https://skyreels-api.oss-accelerate.aliyuncs.com/examples/talking_avatar_video/single_actor/huahai_5s.mp3"

torchrun --nproc_per_node 4 generate_video.py \
    --task_type audio2video_single \
    --model_id $MODEL_PATH \
    --prompt "$PROMPT" \
    --seed 42 \
    --use_usp \
    --offload \
    --quant \
    --input_image "$INPUT_IMAGE" \
    --input_audio "$INPUT_AUDIO"