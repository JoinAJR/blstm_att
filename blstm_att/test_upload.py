'''
Author: your name
Date: 2020-10-15 10:57:24
LastEditTime: 2020-10-15 11:00:29
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \blstm_att\blstm_att\test_upload.py
'''
import paramiko
def ssh_scp_put(ip, port, user, password, local_file, remote_file):
    """

    :param ip: 服务器ip地址
    :param port: 端口(22)
    :param user: 用户名
    :param password: 用户密码
    :param local_file: 本地文件地址
    :param remote_file: 要上传的文件地址（例：/test.txt）
    :return:
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, port, user, password)

    sftp = ssh.open_sftp()
    sftp.put(local_file, remote_file)

ssh_scp_put("119.45.226.43",22,"root","2872451.dai!!","/home/blstm_att/procedure.txt","/home/procedure.txt")