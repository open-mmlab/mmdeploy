pipeline {
    agent { label 'deploy_linux' }

    stages {
        stage('Build') { 
            steps {
                sh ./mmdeploy/tests/jenkins/scripts/docker_run_for_build.sh
            }
        }

    }
  }



