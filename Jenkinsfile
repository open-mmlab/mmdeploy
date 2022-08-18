pipeline {
    agent { label 'deploy_linux' }

    parameters {
        string(
            name: 'CODEBASES', 
            defaultValue: 'mmdet mmcls', 
            description: 'select codebase'
        )
        choice(
            name: 'DOCKER_IMAGE', 
            choices: [
                'mmdeploy-ci-ubuntu-18.04', 
                'mmdeploy-ci-ubuntu-18.04-cu102', 
                'mmdeploy-ci-ubuntu-20.04',
                'mmdeploy-ci-ubuntu-20.04-cu113'
            ], 
            description: 'Pick env'
        )
    }

    stages {
        stage('Build') { 
            steps {
                sh "tests/jenkins/scripts/test_build.sh '${params.DOCKER_IMAGE}'"
            }
        }
        
        stage('Convert') {
            steps {
                sh "tests/jenkins/scripts/test_convert.sh '${params.DOCKER_IMAGE}' '${params.CODEBASES}'"
            }
        }
    }
  }



