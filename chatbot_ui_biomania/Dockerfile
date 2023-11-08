# 基础镜像
FROM node:19-alpine as build

# 设置工作目录
WORKDIR /app

# 复制package.json和package-lock.json
COPY package*.json ./

# 安装依赖
RUN npm install

# 复制其他源文件
COPY . .

EXPOSE 3000

# 启动nginx
CMD ["npm", "run", "dev"]
