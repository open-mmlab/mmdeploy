pipeline {
  agent { label 'deploy_linux' }

  parameters {
    text(
        name: 'release_note', 
        defaultValue: 'Release Note 信息如下所示: \n \
Bug-Fixed: \n \
Feature-Added: ', 
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
                script {
                    String[] codebase_list = ["mmdet", "mmcls"]
                    for (i in codebase_list) {
                        println(i)
                    }
                }
            }
        }

    }
  }
