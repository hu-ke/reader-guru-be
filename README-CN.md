<p align="center">
    <img src="https://hukepublicbucket.oss-cn-hangzhou.aliyuncs.com/readerguru/readerguru-logo.png" />
</p>
<p align="center">
    <br> <a href="README.md">English</a> | 中文</a>
</p>

# 项目介绍
利用**OpenAI**的能力，实现了文档内容**概要生成**和基于文档内容进行**问答**的功能。文件支持`pdf`和`epub`两种文件格式，支持概要和对话的本地持久化。网站免费使用，无需注册或登录。本项目是网站的前端部分代码。[进入网站>>](http://reader.guru/introduction)
# 技术栈
- [前端](https://github.com/hu-ke/reader-guru-fe/)
    - Reactjs v18、dexie v4、Typescript v5
- 服务端
    - python v3.12、gunicorn、fastapi、langchain
# 网站部分截图
文件上传页

![](https://hukepublicbucket.oss-cn-hangzhou.aliyuncs.com/readerguru/readerguru-uploadpage.png)

文件详情页

![](https://hukepublicbucket.oss-cn-hangzhou.aliyuncs.com/readerguru/readerguru-detailpage.png)
# 工作流程
每一次的总结或对话流程最长可能需要等待几分钟，这主要是由上传文件的大小和OpenAI的处理效率决定的。 我们有必要了解下整个过程发生了什么。 以下是工作过程图：

![](https://hukepublicbucket.oss-cn-hangzhou.aliyuncs.com/readerguru/readerguru-flow.png)

1. 在我们开始之前，您需要准备一个`.pdf`或`.epub`格式的文件 如果您没有文件，您可以点击下载一个[示例pdf文件](https://hukepublicbucket.oss-cn-hangzhou.aliyuncs.com/readerguru/IntoThinAirBook.pdf)。上传完成后，服务端会帮您处理剩下的事，您只需要耐心等待。 如果您不是开发人员，可以跳过剩下的步骤。
2. 服务端会从上传的文件中提取所有的文本内容，然后调用合适的`Text Splitter`将文本内容分割成许多独立的`document`对象。
3. 生成`Embeddings`对象，通过利用`OpenAI embedding`。
4. 通过`Embedding`或`Pinecone`来生成`vectors`。
5. 服务器会根据用户操作来生成答案或总结。

# license

[LICENSE](./LICENSE)