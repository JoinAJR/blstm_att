# step1: 基础镜像使用tensorflow-gpu，当然，你也可以使用python作为基础镜像，后面再安装tensorflow-gpu的依赖
FROM python:3.6

# step2: 将工程下面的机器学习相关的文件（这里是mnist文件夹）复制到容器某个目录中，例如：/home/mnist
COPY ./blstm_att /home/blstm_att

# step3 设置容器中的工作目录，直接切换到/home/mnist目录下
WORKDIR /home/blstm_att

# step4 安装依赖
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r requirements.txt
RUN chmod +x /home/blstm_att/runshell.sh

# step5 设置容器启动时的运行命令，这里我们直接运行python程序
CMD ["/home/blstm_att/runshell.sh"]