pipeline {
    agent { label 'deploy_linux' }

    stages {
        stage('Build') { 
            steps {
                cd ./mmdeploy/tests/jenkins/scripts
                sh docker_run_for_build.sh
            }
        }

    }
  }



