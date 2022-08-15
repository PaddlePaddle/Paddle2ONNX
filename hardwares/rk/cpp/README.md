# RKNPU2 c++部署环境搭建

## 软件栈要求

OS：Ubuntu18.04  

rkpu2版本: 
[rknpu2仓库地址](https://github.com/rockchip-linux/rknpu2)

交叉编译工具版本:
gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu

## 编译环境安装
### 安装python环境
请参考python目录下的环境配置

### 安装编译环境

- 下载rknpu,交叉编译工具，放在~/opt目录下
- 修改lib默认配置
```text
sudo gedit /etc/ld.so.conf
# 新增一行
/home/toybrick/opt/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu/lib
```

