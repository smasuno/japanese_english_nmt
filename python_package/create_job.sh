EXECUTOR_IMAGE_URI="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest"

gcloud ai custom-jobs create \
  --region=us-west1 \
  --display-name=ja-en-nmt \
  --worker-pool-spec=machine-type=n1-standard-8,replica-count=1,executor-image-uri=$EXECUTOR_IMAGE_URI,accelerator-type=NVIDIA_TESLA_V100,accelerator-count=1,local-package-path=".",script="run.sh"