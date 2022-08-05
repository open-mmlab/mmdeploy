pipeline {
  agent any

  parameters {
    choice(
      description: '你需要选择哪个模块进行构建 ?',
      name: 'modulename',
      choices: ['Module1', 'Module2', 'Module3']
    )
    
    string(
        description: '你需要在哪台机器上进行部署 ?',
        name: 'deploy_hostname', 
        defaultValue: 'host131', 
    )

    text(
        name: 'release_note', 
        defaultValue: 'Release Note 信息如下所示: \n \
Bug-Fixed: \n \
Feature-Added: ', 
        description: 'Release Note的详细信息是什么 ?'
    )

    booleanParam(
        name: 'test_skip_flag', 
        defaultValue: true, 
        description: '你需要在部署之前执行自动化测试么 ?'
    )


    password(
        name: 'deploy_password', 
        defaultValue: 'liumiaocn', 
        description: '部署机器连接时需要用到的密码信息是什么 '
    )

    file(
        name: "deploy_property_file", 
        description: "你需要输入的部署环境的设定文件是什么 ?"
    )
  }

  stages {
        stage('Build') { 
            steps { 
                echo "Build stage: 选中的构建Module为 : ${params.modulename} ..." 
            }
        }
        stage('Test'){
            steps {
                echo "Test stage: 是否执行自动化测试: ${params.test_skip_flag} ..."
            }
        }
        stage('Deploy') {
            steps {
                echo "Deploy stage: 部署机器的名称 : ${params.deploy_hostname} ..." 
                echo "Deploy stage: 部署连接的密码 : ${params.deploy_password} ..." 
                echo "Deploy stage: Release Note的信息为 : ${params.release_note} ..." 
            }
        }
    }
  }
