#!/bin/bash

if [[ "$(whoami)" == "kristijan" ]]; then
	docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
		--name kbartol-gender \
		-v /home/kristijan/phd/datasets/people3d/:/data/people3d/ \
		-v /home/kristijan/phd/datasets/h36m/:/data/h36m/ \
		-v /home/kristijan/phd/pose/digging-into-3d-from-2d-pose/checkpoint/:/checkpoint \
		-v /home/kristijan/phd/pose/digging-into-3d-from-2d-pose/:/digging-into-3d-from-2d-pose \
		--rm -it kbartol-digging /bin/bash
else
	docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
		--name kbartol-gender \
		-v /home/dbojanic/data/:/data/ \
		-v /home/dbojanic/gender-classifier/:/gender-classifier/ \
		--rm -it kbartol-gender /bin/bash
fi

