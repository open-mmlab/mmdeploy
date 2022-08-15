pipeline {
  agent { label 'deploy_linux' }

  parameters {
    text(
        name: 'codebase_list', 
        defaultValue: 'select codebase', 
        description: '选择codebase'
    )

    booleanParam(
        name: 'mmcls', 
        defaultValue: true, 
    )

    booleanParam(
        name: 'mmdet', 
        defaultValue: true, 
    )

    booleanParam(
        name: 'mmedit', 
        defaultValue: true, 
    )

    booleanParam(
        name: 'mmocr', 
        defaultValue: true, 
    )

    booleanParam(
        name: 'mmpose', 
        defaultValue: true, 
    )

    booleanParam(
        name: 'rotate', 
        defaultValue: true, 
    )

    booleanParam(
        name: 'mmseg', 
        defaultValue: true, 
    )

  }



  stages {
        stage('Build') { 
            steps {
                echo "start build"
                sh """
                    codebase=() \
                    if (( $params.mmcls==true )); then codebase+=(mmcls);fi \ 
                    if (( $params.mmdet==true )); then codebase+=(mmdet);fi \
                    if (( $params.mmedit==true )); then codebase+=(mmedit);fi \
                    if (( $params.mmocr==true )); then codebase+=(mmocr);fi \
                    if (( $params.mmpose==true )); then codebase+=(mmpose);fi \
                    if (( $params.mmrotate==true )); then codebase+=(rotate);fi \
                    if (( $params.mmseg==true )); then codebase+=(mmseg);fi
                """

                sh """
                    for i in {0..6};
                    do
                        echo ${codebase}[i];
                    done
                """ 
            }
        }

    }
  }



