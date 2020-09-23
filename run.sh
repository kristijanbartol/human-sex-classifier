#!/bin/bash

if [[ "$(whoami)" == "kristijan" ]]; then
	docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
		--name kbartol-digging \
		-v /home/kristijan/phd/datasets/people3d/:/data/people3d/ \
		-v /home/kristijan/phd/datasets/h36m/:/data/h36m/ \
		-v /home/kristijan/phd/pose/digging-into-3d-from-2d-pose/checkpoint/:/checkpoint \
		-v /home/kristijan/phd/pose/digging-into-3d-from-2d-pose/:/digging-into-3d-from-2d-pose \
		--rm -it kbartol-digging /bin/bash
else
	docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
		--name kbartol-digging \
		-v /home/dbojanic/data/h36m-fetch/:/data/h36m/ \
		-v /home/dbojanic/digging-into-3d-from-2d-pose/checkpoint/:/checkpoint \
		-v /home/dbojanic/digging-into-3d-from-2d-pose/:/digging-into-3d-from-2d-pose \
		-v /home/dbojanic/data/people3d/:/data/people3d/ \
		--rm -it kbartol-digging /bin/bash
fi

