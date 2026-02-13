# WHartTest - AI驱动的智能测试用例生成平台

## 项目简介

WHartTest 是一个基于 Django REST Framework 构建的AI驱动测试自动化平台，核心功能是通过AI智能生成测试用例。平台集成了 LangChain、MCP（Model Context Protocol）工具调用、项目管理、需求评审、测试用例管理以及先进的知识库管理和文档理解功能。利用大语言模型和多种嵌入服务（OpenAI、Azure OpenAI、Ollama等）的能力，自动化生成高质量的测试用例，并结合知识库提供更精准的测试辅助，为测试团队提供一个完整的智能测试管理解决方案。


## 文档
详细文档请访问：https://mgdaaslab.github.io/WHartTest/



## 快速开始

### ./ai_install.sh 部署（AI智能部署）

注意！！！win环境需要在 Open Git Bash hrre 中打开

<img width="369" height="108" alt="image" src="https://github.com/user-attachments/assets/3d4429a1-395d-4724-b58c-2367cd13ea11" />


```bash
# 1. 克隆仓库
git clone https://github.com/MGdaasLab/WHartTest.git
cd WHartTest

# 2. 准备配置（配置大模型的url和key还有模型）
cp .env.example .env

# 3. 一键启动（与 AI 对话，帮你智能部署）
./ai_install.sh

# 4. 访问系统
# http://localhost:8913 (admin/admin123456)
```

### Docker 部署（推荐 - 开箱即用）

```bash
# 1. 克隆仓库
git clone https://github.com/MGdaasLab/WHartTest.git
cd WHartTest

# 2. 准备配置（使用默认配置，包含自动生成的API Key）
cp .env.example .env

# 3. 一键启动（自动拉取预构建镜像）
docker-compose up -d

# 4. 访问系统
# http://localhost:8913 (admin/admin123456)
```

**就这么简单！** 系统会自动创建默认API Key，MCP服务开箱即用。

> ⚠️ **生产环境提示**：请登录后台删除默认API Key并创建新的安全密钥。详见 [快速启动指南](./docs/QUICK_START.md)

详细的部署说明请参考：
- [快速启动指南](./docs/QUICK_START.md) - **推荐新用户阅读**
- [GitHub 自动构建部署指南](./docs/github-docker-deployment.md)
- [完整部署文档](https://mgdaaslab.github.io/WHartTest/)

## 页面展示

| | |
  |---|---|
  | ![alt text](./img/image.png) | ![alt text](./img/image-1.png) |
  | ![alt text](./img/image-2.png)| ![alt text](./img/image-3.png) |
  | ![alt text](./img/image-4.png) | ![alt text](./img/image-5.png) |
  | ![alt text](./img/image-6.png) | ![alt text](./img/image-7.png) |
  | ![alt text](./img/image-8.png) | ![alt text](./img/image-9.png) |
  | ![alt text](./img/image-10.png) | ![alt text](./img/image-11.png) |
## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request


## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 项目讨论区
- 添加微信时请备注github，拉你进微信群聊。
<img width="400" alt="image" src="https://github.com/user-attachments/assets/b72b71ee-ff6e-4595-9bb1-c02e61951429" />

qq群：

1. 8xxxxxxxx0（已满）
2. 1017708746
---

**WHartTest** - AI驱动测试用例生成，让测试更智能，让开发更高效！

## 【重要安全警示】关于 v1.4.0 版本 Skills 权限及部署安全的声明

鉴于 v1.4.0 版本中 Skills 模块具备较高的系统执行权限，为了保障您的数据与环境安全，我们做出以下严正提示：

部署建议：强烈建议仅在内网环境或受信任的私有网络中部署使用。
访问控制：切勿将服务直接暴露于公网（Public Internet），或授予任何未经身份验证及不可信人员访问权限。
免责声明：本项目（WHartTest）仅供学习与研究使用。用户需自行承担因违规部署（如开放公网、未做鉴权等）所导致的一切安全风险与后果。对于因不当配置引发的数据泄露、服务器被入侵等安全事故，WHartTest 团队不承担任何法律及连带责任。
