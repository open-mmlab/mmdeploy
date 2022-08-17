pipeline {
    agent { label 'deploy_linux' }

    stages {
        stage('Build') { 
            steps {
                sh "tests/jenkins/scripts/docker_run_for_build.sh"
            }
        }

    }
  }



