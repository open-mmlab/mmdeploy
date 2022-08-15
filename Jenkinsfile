pipeline {
  agent { label 'deploy_linux' }

  parameters {
    text(
        name: 'codebase_list', 
        defaultValue: 'select codebase', 
        description: '选择codebase'
    )
    codebase {

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
  }



  stages {
        stage('Build') { 
            steps {
                echo "start build"
                sh """
                    if (( $params.codebase.mmdet==true )); then
                        echo mmdet
                    fi
                """
            }
        }

    }
  }



