Invoke-WebRequest -Uri https://github.com/irexyc/mmdeploy-ci-resource/releases/download/opencv/opencv-win-amd64-4.5.5-vc16.zip -OutFile opencv.zip
Expand-Archive opencv.zip .
Move-Item opencv-4.5.5 opencv
