pipeline {
    agent { label 'deploy_linux' }

    environment {
        docker_image="mmdeploy-ci-ubuntu-18.04"
    }


    stages {
        stage('Build') { 
            steps {
                sh """
                    docker build tests/jenkins/docker/${docker_image}/ -t ${docker_image}
                    docker run /v tests/jenkins/scripts:/root/workspace/scripts -t ${docker_image}
                    container_id=${docker ps | grep ${docker_image} | awk -F ' ' '{print $1}'}
                    docker exec ${container_id} "sh /root/workspace/scripts/docker_exec_for_build.sh"
                """
            }
        }

    }
  }



