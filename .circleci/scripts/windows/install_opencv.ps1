Invoke-WebRequest -Uri https://download.openmmlab.com/mmdeploy/library/opencv-4.5.5.zip -OutFile opencv.zip
Expand-Archive opencv.zip .
Move-Item opencv-4.5.5 opencv
