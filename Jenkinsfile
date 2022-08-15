pipeline {
  agent { label 'deploy_linux' }

  parameters {
    text(
        name: 'codebase_list', 
        defaultValue: 'select codebase', 
        description: 'Release Note的详细信息是什么 ?'
    )

    booleanParam(
        name: 'mmdet', 
        defaultValue: true, 
    )

    booleanParam(
        name: 'mmcls', 
        defaultValue: true, 
    )

  }



  stages {
        stage('Build') { 
            steps {
                echo "start build"
                sh """
                    if [ $params.mmdet == true ]; then
                        echo "mmdet
                    fi
                """
            }
        }

    }
  }
