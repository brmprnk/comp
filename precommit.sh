apptainer exec --writable-tmpfs --pwd /opt/app --containall \
                ./comp.sif pixi -e dev run pre-commit
